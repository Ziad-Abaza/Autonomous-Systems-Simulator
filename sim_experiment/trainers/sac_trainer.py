"""
SAC external trainer entrypoint. Usage:

    python -m sim_experiment.trainers.sac_trainer --run-dir <run_dir>

Continuous-action Soft Actor-Critic. Reads contract.json, trains via
sim_client.agents.sac_baseline.SACRunner, writes metrics.jsonl,
checkpoints/policy_*.pt, periodic evaluation results + replays,
trajectories/, curriculum_state.json, run_result.json.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from sim_experiment.metrics import MetricsWriter
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.trajectory import TrajectoryWriter
from sim_experiment.trainers._harness import (
    write_result, load_contract, build_envs_from_contract, build_eval_env,
    resolve_resume_checkpoint, action_bounds, env_action_type, obs_to_vec,
    probe_obs, setup_curriculum, write_curriculum_state,
    EpisodeTrajectoryRecorder, run_periodic_eval,
)
from sim_experiment.vec_env import is_vec_env


def run_training(run_dir: str) -> int:
    contract, err = load_contract(run_dir)
    if contract is None:
        write_result(run_dir, "failed", error={"type": "invalid_contract", "message": err})
        return 2

    training = contract["training"]
    paths = contract["paths"]
    seed = int(contract["seed"])

    if training.get("algorithm") != "sac":
        write_result(run_dir, "failed",
                     error={"type": "unsupported_algorithm",
                            "message": f"sac_trainer cannot run algorithm "
                                       f"'{training.get('algorithm')}'"})
        return 2
    if env_action_type(contract) != "continuous":
        write_result(run_dir, "failed",
                     error={"type": "incompatible_environment",
                            "message": "SAC requires a continuous action space"})
        return 2

    resume_ckpt = resolve_resume_checkpoint(contract)
    curriculum, curr_err = setup_curriculum(contract, run_dir, resume_ckpt)
    if curr_err:
        write_result(run_dir, "failed", error=curr_err)
        return 2

    writer = MetricsWriter(paths["metrics_file"])
    registry = ArtifactRegistry(run_dir)
    alg = training.get("algorithm_config", {})
    traj = TrajectoryWriter(paths["trajectories_dir"]) if int(alg.get("trajectory_episodes", 0)) > 0 else None
    traj_limit = int(alg.get("trajectory_episodes", 0))

    if curriculum is not None:
        envs = build_envs_from_contract(
            contract, scenario_dict=curriculum.stage_scenario_dict(),
            seed_fn=curriculum.stage_seed)
    else:
        envs = build_envs_from_contract(contract)
    num_envs = len(envs)

    obs_sample = probe_obs(envs, seed)
    obs_dim = obs_to_vec(obs_sample, 0).shape[0]
    act_low, act_high = action_bounds(contract)
    act_dim = int(act_low.shape[0])

    from sim_client.agents.sac_baseline import SACRunner
    runner = SACRunner(
        env=envs,
        obs_dim=obs_dim, act_dim=act_dim,
        action_low=act_low, action_high=act_high,
        lr=float(training.get("learning_rate", 3e-4)),
        gamma=float(training.get("discount", 0.99)),
        buffer_size=int(alg.get("buffer_size", 100_000)),
        warmup_steps=int(alg.get("warmup_steps", 1_000)),
        batch_size=int(training.get("batch_size", 256)),
        tau=float(alg.get("tau", 0.005)),
        alpha_init=float(alg.get("alpha", 0.2)),
        auto_entropy=bool(alg.get("auto_entropy_tuning", True)),
        target_entropy=alg.get("target_entropy"),
        updates_per_step=int(alg.get("updates_per_step", 1)),
        target_update_interval=int(alg.get("target_update_interval", 1)),
        update_interval=int(alg.get("update_interval", 500)),
        seed=seed,
        resume_checkpoint=resume_ckpt,
    )

    recorder = EpisodeTrajectoryRecorder(traj, traj_limit, contract)
    recorder.attach(num_envs)
    runner.on_step = recorder

    ckpt_freq = int(training.get("checkpoint_frequency", 2048))
    eval_freq = int(training.get("eval_frequency", 8192))
    eval_on = bool(contract["evaluation"].get("enabled", True))
    state = {"last_ckpt": 0, "last_eval": 0, "runner": runner}

    def on_update(stats: Dict[str, Any]) -> None:
        step = int(stats["global_step"])

        if curriculum is not None:
            curriculum.record_training_episodes(len(stats["episodes"]))

        for ep in stats["episodes"]:
            writer.write("episode", timestep=step, metrics={
                "reward": ep["return"], "length": ep["length"],
                "env_idx": ep["env_idx"],
                "termination_reason": ep["termination_reason"],
                "checkpoints_passed": ep["checkpoints_passed"],
                "mean_lateral_error": ep["mean_lateral_error"],
                "mean_speed": ep["mean_speed"],
                "collided": ep["collided"], "off_road": ep["off_road"],
            })

        writer.write("step", timestep=step, metrics={
            "sps": stats["sps"], "actor_loss": stats["actor_loss"],
            "critic_loss": stats["critic_loss"], "alpha": stats["alpha"],
            "buffer_size": stats["buffer_size"],
            "episodes_completed": stats["episodes_completed"],
        })

        if ckpt_freq > 0 and step - state["last_ckpt"] >= ckpt_freq:
            state["last_ckpt"] = step
            ckpt_path = os.path.join(paths["checkpoints_dir"], f"policy_{step}.pt")
            state["runner"].save_checkpoint(
                ckpt_path, step=step,
                extra={"curriculum_state": curriculum.to_state()} if curriculum else None)
            registry.register("checkpoint", os.path.relpath(ckpt_path, run_dir), step=step,
                              metadata={
                                  "algorithm": "sac",
                                  "experiment_id": contract["experiment_id"],
                                  "run_id": contract["run_id"],
                                  "env_fingerprint": contract["environment_fingerprint"],
                                  **({"curriculum_stage_index": curriculum.stage_index,
                                      "curriculum_stage_id": curriculum.current_stage.stage_id}
                                     if curriculum and curriculum.current_stage else {}),
                              })

        if eval_on and eval_freq > 0 and step - state["last_eval"] >= eval_freq:
            state["last_eval"] = step
            if curriculum is not None:
                eval_env = build_eval_env(
                    contract, scenario_dict=curriculum.stage_scenario_dict(),
                    seed_fn=curriculum.stage_seed)
            else:
                eval_env = build_eval_env(contract)
            act_fn = make_eval_policy(state["runner"])
            aggregate = run_periodic_eval(
                contract, eval_env, act_fn, step, paths, registry, writer,
                algorithm="sac")
            if curriculum is not None:
                decision = curriculum.evaluate_advancement(aggregate, timestep=step)
                writer.write("curriculum", timestep=step, metrics={
                    "stage_index": curriculum.stage_index,
                    "stage_id": curriculum.current_stage.stage_id if curriculum.current_stage else None,
                    "stage_name": curriculum.current_stage.name if curriculum.current_stage else None,
                    "episodes_in_stage": curriculum.episodes_in_stage,
                    "is_complete": curriculum.is_complete,
                    "decision": decision,
                })
                write_curriculum_state(run_dir, curriculum)
                if decision["advanced"]:
                    if is_vec_env(envs):
                        envs.set_scenario(curriculum.stage_scenario_dict())
                        state["runner"].set_envs(envs)
                    else:
                        new_envs = build_envs_from_contract(
                            contract, scenario_dict=curriculum.stage_scenario_dict(),
                            seed_fn=curriculum.stage_seed)
                        state["runner"].set_envs(new_envs)
                    recorder.attach(num_envs)

    runner.on_update = on_update

    metrics = runner.train(total_timesteps=int(training.get("total_timesteps", 50_000)))

    final_ckpt = os.path.join(paths["checkpoints_dir"], "policy_final.pt")
    runner.save_checkpoint(
        final_ckpt, step=metrics.get("total_timesteps", 0),
        extra={"curriculum_state": curriculum.to_state()} if curriculum else None)
    registry.register("checkpoint", os.path.relpath(final_ckpt, run_dir),
                      step=metrics.get("total_timesteps", 0),
                      metadata={"algorithm": "sac", "final": True,
                                "experiment_id": contract["experiment_id"],
                                "run_id": contract["run_id"],
                                "env_fingerprint": contract["environment_fingerprint"]})
    writer.write("run", timestep=metrics.get("total_timesteps", 0), metrics={
        "total_timesteps": metrics.get("total_timesteps", 0),
        "episodes_completed": metrics.get("episodes_completed", 0),
        "mean_return": metrics.get("overall_mean_return", 0.0),
        "mean_length": metrics.get("overall_mean_length", 0.0),
        "collision_rate": metrics.get("collision_rate", 0.0),
        "off_road_rate": metrics.get("off_road_rate", 0.0),
        "wall_clock_time": metrics.get("wall_clock_time", 0.0),
        "sps": metrics.get("sps", 0.0),
    })

    writer.close()
    if traj is not None:
        traj.close()
    if curriculum is not None:
        write_curriculum_state(run_dir, curriculum)

    write_result(run_dir, "completed",
                 total_timesteps=metrics.get("total_timesteps", 0),
                 episodes_completed=metrics.get("episodes_completed", 0),
                 wall_clock_time=metrics.get("wall_clock_time", 0.0),
                 sps=metrics.get("sps", 0.0),
                 curriculum=curriculum.to_state() if curriculum else None)
    return 0


def make_eval_policy(runner):
    """Frozen deterministic policy for evaluation (tanh mean -> env bounds)."""
    def act(obs) -> np.ndarray:
        vec = obs_to_vec(obs, runner.obs_dim)
        with torch.no_grad():
            a = runner.actor.deterministic(
                torch.tensor(vec, dtype=torch.float32, device=runner.device).unsqueeze(0))
        return runner._to_env_action(a.squeeze(0).cpu().numpy())
    return act


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return run_training(parser.parse_args().run_dir)


if __name__ == "__main__":
    sys.exit(main())
