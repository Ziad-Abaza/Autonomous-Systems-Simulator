"""
Shared external-trainer harness.

Contract-side helpers used by every trainer in sim_experiment.trainers.*:
environment construction, resume-checkpoint resolution, action bounds,
curriculum setup/persistence, trajectory episode recording, and the
periodic evaluation + replay-capture block. Keeps algorithm trainers thin
and guarantees identical contract behavior across PPO/SAC/DQN.
"""

from __future__ import annotations
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.curriculum_runtime import (
    CurriculumController, validate_curriculum,
)
from sim_experiment.evaluation import evaluate_policy
from sim_experiment.headless import build_env_from_dicts
from sim_experiment.metrics import MetricsWriter
from sim_experiment.trainer_contract import validate_contract
from sim_env.curriculum import CurriculumDefinition


def write_result(run_dir: str, status: str, **kwargs) -> None:
    path = os.path.join(run_dir, "run_result.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"status": status, **kwargs}, f, indent=2)
    os.replace(tmp, path)


def build_envs_from_contract(
    contract: Dict[str, Any],
    scenario_dict: Optional[Dict[str, Any]] = None,
    seed_fn: Optional[Callable[[int], int]] = None,
) -> List[Any]:
    """
    Builds num_envs independent environments per contract env_mode.
    `scenario_dict` overrides the contract scenario (curriculum stages);
    `seed_fn(env_index)` overrides the default seed+i derivation.
    """
    training = contract["training"]
    num_envs = max(1, int(training.get("num_envs", 1)))
    seed = int(contract["seed"])
    seed_fn = seed_fn or (lambda i: seed + i)

    if scenario_dict is None:
        scenario_dict = contract.get("scenario") or None
    if scenario_dict is None and os.path.exists(contract["paths"]["scenario_json"]):
        with open(contract["paths"]["scenario_json"], "r", encoding="utf-8") as f:
            scenario_dict = json.load(f)

    if contract["env_mode"] in ("tcp", "tcp_multi"):
        from sim_client.gym_env import SimGymEnv
        from sim_experiment.vec_env import SyncVectorEnv
        host = contract["tcp"]["host"]
        ports = contract["tcp"]["ports"]
        # Wrapped as a VectorEnv so curriculum stage transitions broadcast
        # SET_SCENARIO uniformly (protocol >= 2.1 headless simulators).
        # tcp_multi: ports may repeat (one process, one port, per-client envs).
        vec = SyncVectorEnv(
            [SimGymEnv(host=host, port=p) for p in ports[:num_envs]])
        # The contract's scenario must actually reach the remote sims —
        # otherwise they run whatever scenario their own project embeds.
        if scenario_dict:
            vec.set_scenario(scenario_dict)
        return vec

    with open(contract["paths"]["environment_json"], "r", encoding="utf-8") as f:
        env_dict = json.load(f)

    if contract["env_mode"] == "process":
        from sim_experiment.process_env import ProcessVectorEnv
        return ProcessVectorEnv(
            env_dict, scenario_dict,
            seeds=[seed_fn(i) for i in range(num_envs)],
        )

    return [
        build_env_from_dicts(env_dict, scenario_dict, seed=seed_fn(i))
        for i in range(num_envs)
    ]


def build_eval_env(
    contract: Dict[str, Any],
    scenario_dict: Optional[Dict[str, Any]] = None,
    seed_fn: Optional[Callable[[int], int]] = None,
) -> Any:
    """
    Builds a single in-process evaluation environment from the contract's
    serialized environment/scenario — independent of env_mode. Evaluation
    always runs in-process so replay capture (`env.vehicle`) and frozen
    determinism are available regardless of the training env backend.
    """
    seed = int(contract["seed"])
    seed_fn = seed_fn or (lambda i: seed + i)
    with open(contract["paths"]["environment_json"], "r", encoding="utf-8") as f:
        env_dict = json.load(f)
    if scenario_dict is None:
        scenario_dict = contract.get("scenario") or None
    if scenario_dict is None and os.path.exists(contract["paths"]["scenario_json"]):
        with open(contract["paths"]["scenario_json"], "r", encoding="utf-8") as f:
            scenario_dict = json.load(f)
    return build_env_from_dicts(env_dict, scenario_dict, seed=seed_fn(0))


def resolve_resume_checkpoint(contract: Dict[str, Any]) -> Optional[str]:
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


def action_bounds(contract: Dict[str, Any]) -> "tuple[np.ndarray, np.ndarray]":
    """
    Resolves continuous action bounds from the contract action schema:
    per-channel min/max -> continuous_low/high lists -> vehicle defaults.
    """
    schema = contract.get("action_schema") or {}
    channels = schema.get("channels") or []
    lows, highs = [], []
    for c in channels:
        if "min" in c and "max" in c:
            lows.append(float(c["min"]))
            highs.append(float(c["max"]))
    if lows:
        return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)
    low = schema.get("continuous_low", [-1.0, 0.0, 0.0])
    high = schema.get("continuous_high", [1.0, 1.0, 1.0])
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def env_action_type(contract: Dict[str, Any]) -> str:
    schema = contract.get("action_schema") or {}
    return str(schema.get("space_type") or schema.get("type") or "continuous")


def obs_to_vec(obs: Any, obs_dim: int) -> np.ndarray:
    if isinstance(obs, dict):
        obs = obs.get("vector", np.zeros(obs_dim, dtype=np.float32))
    return np.asarray(obs, dtype=np.float32)


def probe_obs(envs: Any, seed: int) -> Any:
    """Single observation for dimension probing — works for env lists and
    VectorEnv backends (reset env 0 with an explicit seed)."""
    from sim_experiment.vec_env import is_vec_env
    if is_vec_env(envs):
        return envs.reset_at(0, seed=seed)
    return envs[0].reset(seed=seed)[0]


# ----------------------------------------------------------------- curriculum

def write_curriculum_state(run_dir: str, controller: CurriculumController) -> None:
    path = os.path.join(run_dir, "curriculum_state.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(controller.to_state(), f, indent=2)
    os.replace(tmp, path)


def setup_curriculum(
    contract: Dict[str, Any],
    run_dir: str,
    resume_ckpt: Optional[str],
) -> "tuple[Optional[CurriculumController], Optional[Dict[str, Any]]]":
    """
    Builds the CurriculumController for a run, restoring from a resume
    checkpoint when present. Returns (controller, error_dict) — exactly one
    is non-None.
    """
    curriculum_dict = contract.get("curriculum")
    if not curriculum_dict:
        return None, None
    seed = int(contract["seed"])
    errors = validate_curriculum(
        curriculum_dict, known_scenario_ids=[contract.get("scenario_id", "")])
    if errors:
        return None, {"type": "invalid_curriculum", "message": "; ".join(errors)}

    controller = CurriculumController(
        CurriculumDefinition.from_dict(curriculum_dict),
        base_seed=seed,
        base_scenario_dict=contract.get("scenario"),
    )

    if resume_ckpt:
        import torch
        payload = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        saved = payload.get("curriculum_state")
        restart = bool((contract.get("resume") or {}).get("restart_curriculum"))
        if saved:
            if saved.get("curriculum_fingerprint") != contract.get("curriculum_fingerprint"):
                return None, {
                    "type": "curriculum_mismatch",
                    "message": "Checkpoint curriculum does not match this "
                               "experiment's curriculum; refusing to resume.",
                }
            controller = CurriculumController.from_state(
                controller.definition, saved,
                base_scenario_dict=contract.get("scenario"))
        elif not restart:
            return None, {
                "type": "curriculum_state_missing",
                "message": "Checkpoint has no curriculum_state; resume "
                           "requires resume.restart_curriculum=true",
            }

    write_curriculum_state(run_dir, controller)
    return controller, None


# ----------------------------------------------------------------- trajectories

class EpisodeTrajectoryRecorder:
    """
    Records per-episode trajectories from runner on_step records.
    Expects rec = {env_idx, obs, action, reward, terminated, truncated, info}.
    """

    def __init__(self, traj_writer, traj_limit: int, contract: Dict[str, Any]):
        self.traj = traj_writer
        self.limit = int(traj_limit)
        self.contract = contract
        self.seed = int(contract["seed"])
        self.env_episode_idx: List[int] = []
        self.env_step_idx: List[int] = []
        self.episodes_done = 0

    def attach(self, num_envs: int) -> None:
        self.env_episode_idx = [0] * num_envs
        self.env_step_idx = [0] * num_envs

    def __call__(self, rec: Dict[str, Any]) -> None:
        env_idx = rec["env_idx"]
        ep_idx = self.env_episode_idx[env_idx]
        self.env_step_idx[env_idx] += 1
        info = rec["info"]
        if self.traj is not None and ep_idx < self.limit:
            ep_id = f"train_env{env_idx}_ep{ep_idx}"
            if self.env_step_idx[env_idx] == 1:
                self.traj.start_episode(
                    ep_id,
                    env_fingerprint=self.contract["environment_fingerprint"],
                    scenario_id=self.contract["scenario_id"],
                    seed=self.seed,
                    observation_schema=self.contract.get("observation_schema"),
                    action_schema=self.contract.get("action_schema"),
                )
            self.traj.record_step(
                step=self.env_step_idx[env_idx],
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
            if self.traj is not None and ep_idx < self.limit:
                self.episodes_done += 1
                self.traj.close_episode()
            self.env_episode_idx[env_idx] += 1
            self.env_step_idx[env_idx] = 0


# ----------------------------------------------------------------- evaluation

def run_periodic_eval(
    contract: Dict[str, Any],
    eval_env: Any,
    act_fn: Callable[[Any], Any],
    step: int,
    paths: Dict[str, str],
    registry: ArtifactRegistry,
    writer: MetricsWriter,
    algorithm: str,
) -> Dict[str, Any]:
    """
    Runs a frozen-policy evaluation in a fresh env (built by the caller,
    honoring the current curriculum stage), persists the result + first-episode
    replay, and returns the aggregate dict.
    """
    eval_cfg = contract.get("evaluation", {})
    seed = int(contract["seed"])

    replay_frames: List[Dict[str, Any]] = []

    def _observe(ep_i: int, step_i: int, action, reward: float, info: Dict[str, Any]) -> None:
        if ep_i != 0 or not hasattr(eval_env, "vehicle"):
            return
        st = eval_env.vehicle.state
        replay_frames.append({
            "step": step_i, "t": round(info.get("sim_time", 0.0), 4),
            "pos": [round(st.pos.x, 3), round(st.pos.y, 3), round(st.pos.z, 3)],
            "yaw": round(st.yaw, 4), "speed": round(st.speed, 2),
            "action": [round(float(a), 3) for a in (action if hasattr(action, "__len__") else [action])],
            "reward": round(reward, 4),
            "breakdown": {k: round(v, 4) for k, v in info.get("reward_breakdown", {}).items()},
            "lat_offset": round(info.get("lateral_offset", 0.0), 3),
            "heading_err": round(info.get("heading_error", 0.0), 4),
            "collision": bool(info.get("is_colliding", False)),
        })

    result = evaluate_policy(
        eval_env, act_fn,
        seeds=eval_cfg.get("eval_seeds", [seed]),
        num_episodes=int(eval_cfg.get("num_episodes", 3)),
        deterministic=True,
        result_kwargs={
            "checkpoint_path": "in_training",
            "algorithm": algorithm,
            "env_fingerprint": contract["environment_fingerprint"],
            "scenario_id": contract["scenario_id"],
        },
        step_observer=_observe,
    )
    eval_path = os.path.join(paths["evaluation_dir"], f"eval_{step}.json")
    result.save(eval_path)
    registry.register("evaluation", os.path.relpath(eval_path, paths["run_dir"]), step=step)
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
        registry.register("replay", os.path.relpath(replay_path, paths["run_dir"]), step=step,
                          metadata={"episode": 0, "evaluation": True})
    return dict(result.aggregate)


def load_contract(run_dir: str) -> "tuple[Optional[Dict[str, Any]], Optional[str]]":
    """Loads + validates contract.json. Returns (contract, error_message)."""
    with open(os.path.join(run_dir, "contract.json"), "r", encoding="utf-8") as f:
        contract = json.load(f)
    errors = validate_contract(contract)
    if errors:
        return None, "; ".join(errors)
    return contract, None
