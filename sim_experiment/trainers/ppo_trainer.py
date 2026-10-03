"""
Contract-driven PPO trainer (external trainer process).

Launched by the orchestrator as:

    python -m sim_experiment.trainers.ppo_trainer --run-dir <run_dir>

Reads contract.json, builds the environment (in-process simulation or TCP
via headless simulators), trains with the PPORunner baseline, and reports
through the contract artifacts: metrics.jsonl, checkpoints/, evaluation/,
trajectories/, and run_result.json.

This module does NOT live in the simulation core — it is an external
client of the environment API. Shared contract-side plumbing lives in
sim_experiment.trainers._harness.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from sim_experiment.metrics import MetricsWriter
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.trajectory import TrajectoryWriter
from sim_experiment.trainers._harness import (
    write_result, load_contract, build_envs_from_contract, build_eval_env,
    resolve_resume_checkpoint, setup_curriculum, write_curriculum_state,
    EpisodeTrajectoryRecorder, run_periodic_eval,
)
from sim_experiment.vec_env import is_vec_env


def run_training(run_dir: str) -> int:
    contract, err = load_contract(run_dir)
    if contract is None:
        write_result(run_dir, "failed",
                     error={"type": "invalid_contract", "message": err})
        return 2

    paths = contract["paths"]
    training = contract["training"]
    alg = training.get("algorithm_config", {})
    seed = int(contract["seed"])

    if training.get("algorithm", "ppo") != "ppo":
        write_result(run_dir, "failed",
                     error={"type": "unsupported_algorithm",
                            "message": f"ppo_trainer cannot run algorithm '{training.get('algorithm')}'"})
        return 2

    resume_ckpt = resolve_resume_checkpoint(contract)
    curriculum, curr_err = setup_curriculum(contract, run_dir, resume_ckpt)
    if curr_err:
        write_result(run_dir, "failed", error=curr_err)
        return 2

    writer = MetricsWriter(paths["metrics_file"])
    registry = ArtifactRegistry(run_dir)
    traj = TrajectoryWriter(paths["trajectories_dir"]) if int(alg.get("trajectory_episodes", 0)) > 0 else None
    traj_limit = int(alg.get("trajectory_episodes", 0))

    if curriculum is not None:
        envs = build_envs_from_contract(
            contract, scenario_dict=curriculum.stage_scenario_dict(),
            seed_fn=curriculum.stage_seed)
    else:
        envs = build_envs_from_contract(contract)
    num_envs = len(envs)

    recorder = EpisodeTrajectoryRecorder(traj, traj_limit, contract)
    recorder.attach(num_envs)

    state = {"last_ckpt": 0, "last_eval": 0, "runner": None, "envs": envs}
    log_freq = max(1, int(training.get("logging_frequency", 1)))
    ckpt_freq = int(training.get("checkpoint_frequency", 0))
    eval_freq = int(training.get("eval_frequency", 0))

    def _policy(obs):
        """Deterministic policy from the in-training ActorCritic (actor mean)."""
        import torch
        x = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        runner = state["runner"]
        with torch.no_grad():
            feat = runner.agent.actor_backbone(x)
            mean = runner.agent.actor_mean(feat)
            a = mean.squeeze(0).cpu().numpy()
        return [float(np.clip(a[0], -1.0, 1.0)),
                float(np.clip(a[1], 0.0, 1.0)),
                float(np.clip(a[2], 0.0, 1.0))]

    def _curriculum_metrics() -> Dict[str, Any]:
        stage = curriculum.current_stage if curriculum else None
        return {
            "stage_index": curriculum.stage_index,
            "stage_id": stage.stage_id if stage else None,
            "stage_name": stage.name if stage else None,
            "episodes_in_stage": curriculum.episodes_in_stage,
            "is_complete": curriculum.is_complete,
        }

    def on_update(stats: Dict[str, Any]) -> None:
        step = int(stats["global_step"])

        if curriculum is not None:
            curriculum.record_training_episodes(len(stats["episodes"]))

        for ep in stats["episodes"]:
            writer.write("episode", timestep=step, metrics={
                "reward": ep["return"],
                "length": ep["length"],
                "termination_reason": ep["termination_reason"],
                "mean_lateral_error": ep["mean_lateral_error"],
                "mean_speed": ep["mean_speed"],
                "checkpoints_passed": ep["checkpoints_passed"],
            })

        if stats["update"] % log_freq == 0:
            writer.write("run", timestep=step, metrics={
                "sps": stats["sps"],
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"],
                "approx_kl": stats["approx_kl"],
                "explained_variance": stats["explained_variance"],
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
                                  "algorithm": "ppo",
                                  "experiment_id": contract["experiment_id"],
                                  "run_id": contract["run_id"],
                                  "env_fingerprint": contract["environment_fingerprint"],
                                  **({"curriculum_stage_index": curriculum.stage_index,
                                      "curriculum_stage_id": curriculum.current_stage.stage_id}
                                     if curriculum and curriculum.current_stage else {}),
                              })

        if eval_freq > 0 and step - state["last_eval"] >= eval_freq:
            state["last_eval"] = step
            if curriculum is not None:
                eval_env = build_eval_env(
                    contract, scenario_dict=curriculum.stage_scenario_dict(),
                    seed_fn=curriculum.stage_seed)
            else:
                eval_env = build_eval_env(contract)

            aggregate = run_periodic_eval(
                contract, eval_env, _policy, step, paths, registry, writer,
                algorithm="ppo")

            # Curriculum advancement consumes the evaluation aggregate —
            # never raw training reward. The decision is always recorded.
            if curriculum is not None:
                decision = curriculum.evaluate_advancement(aggregate, timestep=step)
                writer.write("curriculum", timestep=step, metrics={
                    **_curriculum_metrics(), "decision": decision,
                })
                write_curriculum_state(run_dir, curriculum)
                if decision["advanced"]:
                    envs = state["envs"]
                    if is_vec_env(envs):
                        # Process workers keep running — broadcast the new
                        # stage scenario; set_envs marks rollout dirty and
                        # reseeds each env deterministically.
                        envs.set_scenario(curriculum.stage_scenario_dict())
                        state["runner"].set_envs(envs)
                    else:
                        new_envs = build_envs_from_contract(
                            contract, scenario_dict=curriculum.stage_scenario_dict(),
                            seed_fn=curriculum.stage_seed)
                        state["runner"].set_envs(new_envs)
                    recorder.attach(num_envs)

    from sim_client.agents.ppo_baseline import PPORunner

    runner = PPORunner(
        env=envs,
        lr=float(training.get("learning_rate", 3e-4)),
        gamma=float(training.get("discount_factor", 0.99)),
        gae_lambda=float(training.get("gae_lambda", 0.95)),
        clip_coef=float(alg.get("clip_coef", 0.2)),
        ent_coef=float(alg.get("ent_coef", 0.01)),
        vf_coef=float(alg.get("vf_coef", 0.5)),
        max_grad_norm=float(alg.get("max_grad_norm", 0.5)),
        num_steps=int(training.get("rollout_length", 1024)),
        num_epochs=int(training.get("epochs", 4)),
        batch_size=int(training.get("batch_size", 256)),
        seed=seed,
        on_update=on_update,
        on_step=recorder if traj is not None else None,
        resume_checkpoint=resume_ckpt,
    )
    state["runner"] = runner

    metrics = runner.train(total_timesteps=int(training.get("total_timesteps", 50000)))

    # Final checkpoint + run-scope summary
    final_ckpt = os.path.join(paths["checkpoints_dir"], "policy_final.pt")
    runner.save_checkpoint(
        final_ckpt, step=metrics.get("total_timesteps", 0),
        extra={"curriculum_state": curriculum.to_state()} if curriculum else None)
    registry.register("checkpoint", os.path.relpath(final_ckpt, run_dir),
                      step=metrics.get("total_timesteps", 0),
                      metadata={"algorithm": "ppo", "final": True,
                                "experiment_id": contract["experiment_id"],
                                "run_id": contract["run_id"],
                                "env_fingerprint": contract["environment_fingerprint"],
                                **({"curriculum_stage_index": curriculum.stage_index,
                                    "curriculum_stage_id": curriculum.current_stage.stage_id}
                                   if curriculum and curriculum.current_stage else {})})
    writer.write("run", timestep=metrics.get("total_timesteps", 0), metrics={
        "final": 1.0,
        "episodes_completed": metrics.get("episodes_completed", 0),
        "overall_mean_return": metrics.get("overall_mean_return", 0.0),
        "collision_rate": metrics.get("collision_rate", 0.0),
        "off_road_rate": metrics.get("off_road_rate", 0.0),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return run_training(parser.parse_args().run_dir)


if __name__ == "__main__":
    sys.exit(main())
