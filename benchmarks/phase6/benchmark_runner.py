"""
Phase 6 learning benchmark runner — measured learning, not vibes.

Phases:
    baseline  — frozen *untrained* policy evaluated on the fixed eval seeds
    phase_i   — train (or resume) for phase timesteps -> checkpoint -> eval

Every evaluation uses `evaluate_policy` + `make_policy_from_checkpoint`
against a freshly built environment on the same seed list, so phase deltas
are comparable. The report records the config hash, sim/torch versions,
platform info, wall time, SPS, and per-seed aggregates — a downstream
`convergence_report` turns it into a verdict.

Usage:
    python benchmarks/phase6/benchmark_runner.py \
        --config benchmarks/phase6/benchmark_config.json [--work-dir DIR]
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _config_hash(config: Dict[str, Any]) -> str:
    """Stable sha256 of the benchmark config (sorted keys, str defaults)."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _environment_info() -> Dict[str, Any]:
    import torch
    import sim_version
    return {
        "sim_version": getattr(sim_version, "VERSION",
                               getattr(sim_version, "__version__", "unknown")),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
    }


def _untrained_policy(algorithm: str, obs_dim: int, seed: int):
    """Freshly-initialized policy for the untrained baseline phase."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    if algorithm == "ppo":
        from sim_client.agents.ppo_baseline import ActorCritic
        act_dim = 3
        net = ActorCritic(obs_dim, act_dim)
        net.eval()

        def policy(obs):
            if isinstance(obs, dict):
                obs = obs.get("vector", np.zeros(obs_dim, dtype=np.float32))
            x = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                feat = net.actor_backbone(x)
                a = net.actor_mean(feat).squeeze(0).cpu().numpy()
            return [float(np.clip(a[0], -1, 1)),
                    float(np.clip(a[1], 0, 1)),
                    float(np.clip(a[2], 0, 1))]
        return policy
    raise ValueError(f"No untrained-policy factory for algorithm '{algorithm}'")


def _eval_policy(env, act_fn, seeds, num_episodes, result_kwargs):
    from sim_experiment.evaluation import evaluate_policy
    return evaluate_policy(
        env, act_fn, seeds=seeds, num_episodes=num_episodes,
        deterministic=True, result_kwargs=result_kwargs)


def _eval_aggregate(result) -> Dict[str, Any]:
    per_seed = {}
    for ep in result.episodes:
        per_seed[str(ep["seed"])] = ep["reward"]
    return {"aggregate": dict(result.aggregate), "per_seed": per_seed}


def run_benchmark(config: Dict[str, Any], work_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the full baseline -> train -> eval -> resume -> eval loop.
    Returns (and persists) the results dict consumed by convergence_report.
    """
    from sim_env.templates import EnvironmentTemplateManager
    from sim_experiment.manifest import (
        ExperimentManifest, TrainingConfig, EvaluationConfig)
    from sim_experiment.manager import ExperimentManager
    from sim_experiment.orchestrator import LocalTrainingOrchestrator
    from sim_experiment.headless import build_env_from_dicts
    from sim_experiment.evaluation import make_policy_from_checkpoint
    from sim_experiment.artifacts import ArtifactRegistry
    from sim_experiment.metrics import MetricsReader

    work_dir = os.path.abspath(
        work_dir or os.path.join("benchmarks", "phase6", "results",
                                 time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(work_dir, exist_ok=True)

    template = config.get("template", "lane_following")
    trainer = config.get("trainer", "ppo")
    seed = int(config.get("random_seed", 42))
    eval_seeds = [int(s) for s in config.get("eval_seeds", [seed])]
    num_eps = int(config.get("num_eval_episodes", 3))
    overrides = dict(config.get("training_overrides") or {})
    phases_cfg = list(config.get("phases") or [])
    env_mode = config.get("env_mode", "inprocess")

    project = EnvironmentTemplateManager.create_project_from_template(template)
    mgr = ExperimentManager(root_dir=os.path.join(work_dir, "experiments"))

    results: Dict[str, Any] = {
        "benchmark_name": config.get("name", "phase6_benchmark"),
        "config": config,
        "config_hash": _config_hash(config),
        "environment": _environment_info(),
        "work_dir": work_dir,
        "phases": [],
        "started_at": time.time(),
    }

    t0 = time.perf_counter()

    # --- Manifest / experiment -------------------------------------------
    training_over = dict(overrides)
    first_phase = phases_cfg[0] if phases_cfg else {}
    total_ts = int(first_phase.get("timesteps", 1000))
    training = TrainingConfig(
        algorithm=trainer,
        total_timesteps=total_ts,
        rollout_length=int(first_phase.get("rollout_length",
                                           training_over.pop("rollout_length", 256))),
        num_envs=int(training_over.pop("num_envs", 1)),
        eval_frequency=int(training_over.pop("eval_frequency", 0)),
        checkpoint_frequency=int(training_over.pop("checkpoint_frequency", 0)),
        batch_size=int(training_over.pop("batch_size", 256)),
        epochs=int(training_over.pop("epochs", 4)),
        algorithm_config=training_over.pop("algorithm_config", {}),
    )
    manifest = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=training, evaluation=EvaluationConfig(eval_seeds=eval_seeds,
                                                       num_episodes=num_eps),
        name=config.get("name", "bench"), random_seed=seed)
    exp_dir = mgr.create(manifest)
    orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)

    # --- Baseline: untrained policy ---------------------------------------
    eval_env = build_env_from_dicts(
        manifest.environment, manifest.scenario_configuration, seed=seed)
    try:
        obs0, _ = eval_env.reset(seed=seed)
        v0 = obs0.get("vector") if isinstance(obs0, dict) else obs0
        import numpy as np
        obs_dim = int(np.asarray(v0, dtype=np.float32).reshape(-1).size)
        base_policy = _untrained_policy(trainer, obs_dim, seed)
        res = _eval_policy(eval_env, base_policy, eval_seeds, num_eps,
                           {"algorithm": trainer})
        results["phases"].append({
            "name": "baseline", "timesteps": 0,
            "eval": _eval_aggregate(res),
        })
    finally:
        if hasattr(eval_env, "close"):
            eval_env.close()

    # --- Training phases ---------------------------------------------------
    prev_run_id: Optional[str] = None
    for i, ph in enumerate(phases_cfg, start=1):
        tstart = time.perf_counter()
        training.total_timesteps = int(ph.get("timesteps", total_ts))
        resume = None
        if prev_run_id:
            # Resume continues training from the prior phase's checkpoint.
            prev_rd = os.path.join(exp_dir, "runs", prev_run_id)
            ck = ArtifactRegistry(prev_rd).latest_of_kind("checkpoint")
            if ck:
                resume = {"parent_run_id": prev_run_id,
                          "checkpoint": os.path.join(prev_rd, ck["path"])}
        run_id = orch.launch(manifest, exp_dir, trainer=trainer,
                             env_mode=env_mode, resume_from=resume)
        summary = orch.wait(exp_dir, run_id, timeout_s=float(
            config.get("phase_timeout_s", 1200)))
        run_wall = time.perf_counter() - tstart
        rd = os.path.join(exp_dir, "runs", run_id)

        ck = ArtifactRegistry(rd).latest_of_kind("checkpoint")
        ckpt_abs = os.path.join(rd, ck["path"]) if ck else None

        eval_env = build_env_from_dicts(
            manifest.environment, manifest.scenario_configuration, seed=seed)
        try:
            if ckpt_abs:
                act_fn = make_policy_from_checkpoint(
                    ckpt_abs, algorithm=trainer, deterministic=True)
            else:
                act_fn = _untrained_policy(trainer, obs_dim, seed)
            res = _eval_policy(eval_env, act_fn, eval_seeds, num_eps,
                               {"algorithm": trainer,
                                "checkpoint_path": ckpt_abs})
            phase_eval = _eval_aggregate(res)
        finally:
            if hasattr(eval_env, "close"):
                eval_env.close()

        run_result = {}
        rr = os.path.join(rd, "run_result.json")
        if os.path.exists(rr):
            with open(rr) as f:
                run_result = json.load(f)

        results["phases"].append({
            "name": f"phase{i}",
            "timesteps": int(ph.get("timesteps", total_ts)),
            "run": {"run_id": run_id, "status": summary["status"]},
            "checkpoint": ckpt_abs,
            "sps": run_result.get("sps"),
            "wall_clock_s": round(run_wall, 2),
            "eval": phase_eval,
        })
        prev_run_id = run_id

    results["wall_clock_s"] = round(time.perf_counter() - t0, 2)
    results["finished_at"] = time.time()

    out = os.path.join(work_dir, "benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    results = run_benchmark(config, work_dir=args.work_dir)

    from sim_experiment.convergence import convergence_report
    report = convergence_report(results)
    print(json.dumps(report, indent=2))
    print(f"Results: {results['work_dir']}/benchmark_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
