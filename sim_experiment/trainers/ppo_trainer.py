"""
Contract-driven PPO trainer (external trainer process).

Launched by the orchestrator as:

    python -m sim_experiment.trainers.ppo_trainer --run-dir <run_dir>

Reads contract.json, builds the environment (in-process simulation or TCP
via headless simulators), trains with the PPORunner baseline, and reports
through the contract artifacts: metrics.jsonl, checkpoints/, evaluation/,
trajectories/, and run_result.json.

This module does NOT live in the simulation core — it is an external
client of the environment API.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from sim_experiment.metrics import MetricsWriter
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.trajectory import TrajectoryWriter
from sim_experiment.trainer_contract import validate_contract
from sim_experiment.evaluation import evaluate_policy
from sim_experiment.headless import build_env_from_dicts


def _write_result(run_dir: str, status: str, **kwargs) -> None:
    path = os.path.join(run_dir, "run_result.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"status": status, **kwargs}, f, indent=2)
    os.replace(tmp, path)


def _build_envs(contract: Dict[str, Any]) -> List[Any]:
    """Builds num_envs independent environments per contract env_mode."""
    training = contract["training"]
    num_envs = max(1, int(training.get("num_envs", 1)))
    seed = int(contract["seed"])

    if contract["env_mode"] == "tcp":
        from sim_client.gym_env import SimGymEnv
        host = contract["tcp"]["host"]
        ports = contract["tcp"]["ports"]
        return [SimGymEnv(host=host, port=p) for p in ports[:num_envs]]

    # inprocess: independent SimulationEnvironment instances
    with open(contract["paths"]["environment_json"], "r", encoding="utf-8") as f:
        env_dict = json.load(f)
    scenario_dict = contract.get("scenario") or None
    if scenario_dict is None and os.path.exists(contract["paths"]["scenario_json"]):
        with open(contract["paths"]["scenario_json"], "r", encoding="utf-8") as f:
            scenario_dict = json.load(f)
    return [
        build_env_from_dicts(env_dict, scenario_dict, seed=seed + i)
        for i in range(num_envs)
    ]


def _resolve_resume_checkpoint(contract: Dict[str, Any]) -> Optional[str]:
    resume = contract.get("resume")
    if not resume:
        return None
    ckpt = resume.get("checkpoint")
    if not ckpt:
        return None
    if os.path.isabs(ckpt) and os.path.exists(ckpt):
        return ckpt
    candidates = [
        os.path.join(contract["paths"]["run_dir"], ckpt),
        os.path.join(
            contract["paths"]["experiment_dir"], "runs",
            resume.get("parent_run_id", ""), ckpt
        ),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_training(run_dir: str) -> int:
    with open(os.path.join(run_dir, "contract.json"), "r", encoding="utf-8") as f:
        contract = json.load(f)
    errors = validate_contract(contract)
    if errors:
        _write_result(run_dir, "failed",
                      error={"type": "invalid_contract", "message": "; ".join(errors)})
        return 2

    paths = contract["paths"]
    training = contract["training"]
    alg = training.get("algorithm_config", {})
    seed = int(contract["seed"])

    if training.get("algorithm", "ppo") != "ppo":
        _write_result(run_dir, "failed",
                      error={"type": "unsupported_algorithm",
                             "message": f"ppo_trainer cannot run algorithm '{training.get('algorithm')}'"})
        return 2

    writer = MetricsWriter(paths["metrics_file"])
    registry = ArtifactRegistry(run_dir)
    traj = TrajectoryWriter(paths["trajectories_dir"]) if int(alg.get("trajectory_episodes", 0)) > 0 else None
    traj_limit = int(alg.get("trajectory_episodes", 0))

    envs = _build_envs(contract)
    num_envs = len(envs)

    # Per-env episode counters for trajectory capture
    env_episode_idx = [0] * num_envs
    env_step_idx = [0] * num_envs
    traj_episodes_done = 0

    def on_step(rec: Dict[str, Any]) -> None:
        nonlocal traj_episodes_done
        env_idx = rec["env_idx"]
        ep_idx = env_episode_idx[env_idx]
        env_step_idx[env_idx] += 1
        info = rec["info"]
        if traj is not None and ep_idx < traj_limit:
            ep_id = f"train_env{env_idx}_ep{ep_idx}"
            if env_step_idx[env_idx] == 1:
                traj.start_episode(
                    ep_id,
                    env_fingerprint=contract["environment_fingerprint"],
                    scenario_id=contract["scenario_id"],
                    seed=seed,
                    observation_schema=contract.get("observation_schema"),
                    action_schema=contract.get("action_schema"),
                )
            traj.record_step(
                step=env_step_idx[env_idx],
                agent_data={
                    "obs": rec["obs"],
                    "action": rec["action"],
                    "reward": rec["reward"],
                    "terminated": rec["terminated"],
                    "truncated": rec["truncated"],
                    "termination_reason": info.get("termination_reason", "running"),
                },
                diagnostic_data={
                    "speed": info.get("speed", 0.0),
                    "lateral_offset": info.get("lateral_offset", 0.0),
                    "heading_error": info.get("heading_error", 0.0),
                    "is_colliding": info.get("is_colliding", False),
                    "is_on_road": info.get("is_on_road", True),
                    "checkpoints_passed": info.get("checkpoints_passed", 0),
                },
            )
        if rec["terminated"] or rec["truncated"]:
            if traj is not None and ep_idx < traj_limit:
                traj_episodes_done += 1
                traj.close_episode()
            env_episode_idx[env_idx] += 1
            env_step_idx[env_idx] = 0

    state = {"last_ckpt": 0, "last_eval": 0, "runner": None}
    log_freq = max(1, int(training.get("logging_frequency", 1)))
    ckpt_freq = int(training.get("checkpoint_frequency", 0))
    eval_freq = int(training.get("eval_frequency", 0))
    eval_cfg = contract.get("evaluation", {})

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

    def on_update(stats: Dict[str, Any]) -> None:
        step = int(stats["global_step"])

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
            state["runner"].save_checkpoint(ckpt_path, step=step)
            registry.register("checkpoint", os.path.relpath(ckpt_path, run_dir), step=step,
                              metadata={
                                  "algorithm": "ppo",
                                  "experiment_id": contract["experiment_id"],
                                  "run_id": contract["run_id"],
                                  "env_fingerprint": contract["environment_fingerprint"],
                              })

        if eval_freq > 0 and step - state["last_eval"] >= eval_freq:
            state["last_eval"] = step
            eval_env = _build_envs(contract)[0]

            # Capture replay frames for the first eval episode only
            replay_frames: List[Dict[str, Any]] = []

            def _observe(ep_i: int, step_i: int, action, reward: float, info: Dict[str, Any]) -> None:
                if ep_i != 0:
                    return
                st = eval_env.vehicle.state
                replay_frames.append({
                    "step": step_i, "t": round(info.get("sim_time", 0.0), 4),
                    "pos": [round(st.pos.x, 3), round(st.pos.y, 3), round(st.pos.z, 3)],
                    "yaw": round(st.yaw, 4), "speed": round(st.speed, 2),
                    "action": [round(float(a), 3) for a in action],
                    "reward": round(reward, 4),
                    "breakdown": {k: round(v, 4) for k, v in info.get("reward_breakdown", {}).items()},
                    "lat_offset": round(info.get("lateral_offset", 0.0), 3),
                    "heading_err": round(info.get("heading_error", 0.0), 4),
                    "collision": bool(info.get("is_colliding", False)),
                })

            result = evaluate_policy(
                eval_env, _policy,
                seeds=eval_cfg.get("eval_seeds", [seed]),
                num_episodes=int(eval_cfg.get("num_episodes", 3)),
                deterministic=True,
                result_kwargs={
                    "checkpoint_path": "in_training",
                    "algorithm": "ppo",
                    "env_fingerprint": contract["environment_fingerprint"],
                    "scenario_id": contract["scenario_id"],
                },
                step_observer=_observe,
            )
            eval_path = os.path.join(paths["evaluation_dir"], f"eval_{step}.json")
            result.save(eval_path)
            registry.register("evaluation", os.path.relpath(eval_path, run_dir), step=step)
            writer.write("evaluation", timestep=step, metrics=dict(result.aggregate))

            if replay_frames:
                replay_path = os.path.join(paths["replays_dir"], f"eval_{step}_ep0.json")
                with open(replay_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "metadata": {
                            "env_fingerprint": contract["environment_fingerprint"],
                            "scenario_name": contract["scenario_id"],
                            "seed": eval_cfg.get("eval_seeds", [seed])[0],
                            "experiment_id": contract["experiment_id"],
                            "run_id": contract["run_id"],
                            "total_steps": len(replay_frames),
                            "termination_reason": result.episodes[0]["termination_reason"] if result.episodes else "",
                        },
                        "frames": replay_frames,
                    }, f)
                registry.register("replay", os.path.relpath(replay_path, run_dir), step=step,
                                  metadata={"episode": 0, "evaluation": True})

    from sim_client.agents.ppo_baseline import PPORunner

    resume_ckpt = _resolve_resume_checkpoint(contract)
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
        on_step=on_step if traj is not None else None,
        resume_checkpoint=resume_ckpt,
    )
    state["runner"] = runner

    metrics = runner.train(total_timesteps=int(training.get("total_timesteps", 50000)))

    # Final checkpoint + run-scope summary
    final_ckpt = os.path.join(paths["checkpoints_dir"], "policy_final.pt")
    runner.save_checkpoint(final_ckpt, step=metrics.get("total_timesteps", 0))
    registry.register("checkpoint", os.path.relpath(final_ckpt, run_dir),
                      step=metrics.get("total_timesteps", 0),
                      metadata={"algorithm": "ppo", "final": True,
                                "experiment_id": contract["experiment_id"],
                                "run_id": contract["run_id"],
                                "env_fingerprint": contract["environment_fingerprint"]})
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

    _write_result(run_dir, "completed",
                  total_timesteps=metrics.get("total_timesteps", 0),
                  episodes_completed=metrics.get("episodes_completed", 0),
                  wall_clock_time=metrics.get("wall_clock_time", 0.0),
                  sps=metrics.get("sps", 0.0))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    rd = os.path.abspath(args.run_dir)
    try:
        return run_training(rd)
    except Exception as e:
        _write_result(rd, "failed",
                      error={"type": "trainer_exception",
                             "message": str(e),
                             "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    sys.exit(main())
