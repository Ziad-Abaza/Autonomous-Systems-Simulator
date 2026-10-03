"""
Training & Experiment Platform CLI.

Thin wrapper over the domain services — the CLI contains no business
logic of its own:

    python -m sim_experiment.cli validate-env --project presets/oval_circuit.sim.json
    python -m sim_experiment.cli create --project ... --scenario basic_lane_following \
        --name exp1 --algorithm ppo --timesteps 50000 --seed 42
    python -m sim_experiment.cli list
    python -m sim_experiment.cli show <experiment_id>
    python -m sim_experiment.cli launch <experiment_id> [--trainer ppo]
    python -m sim_experiment.cli batch <experiment_id> --seeds 1 2 3
    python -m sim_experiment.cli runs <experiment_id>
    python -m sim_experiment.cli status <experiment_id> <run_id> [--watch]
    python -m sim_experiment.cli cancel <experiment_id> <run_id>
    python -m sim_experiment.cli resume <experiment_id> <run_id>
    python -m sim_experiment.cli evaluate <experiment_id> <run_id> [--checkpoint PATH]
    python -m sim_experiment.cli trajectories <experiment_id> <run_id>
    python -m sim_experiment.cli reproduce <experiment_id>
    python -m sim_experiment.cli export <experiment_id> --dest PATH
    python -m sim_experiment.cli archive <experiment_id>
    python -m sim_experiment.cli benchmark [--envs 1,2,4] [--steps 2000]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import List, Optional

from sim_project.serializer import EnvironmentProject
from sim_env.scenario_designer import ScenarioDefinition
from sim_env.validator import EnvironmentValidator
from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.run import RunManager, RunStatus
from sim_experiment.orchestrator import LocalTrainingOrchestrator
from sim_experiment.metrics import MetricsReader
from sim_experiment.batch import expand_run_specs
from sim_experiment.reproduce import check_reproducibility
from sim_experiment.headless import build_env_from_dicts, HeadlessEnvPool
from sim_experiment.evaluation import evaluate_policy, make_policy_from_checkpoint
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.capabilities import TRAINER_CAPABILITIES
from sim_experiment.scheduler import BatchScheduler, RetryPolicy
from sim_experiment.analytics import compare_runs, compare_experiments, list_run_dirs
from sim_experiment.dataset import export_dataset, list_episodes


def _mgr(args) -> ExperimentManager:
    return ExperimentManager(root_dir=args.root)


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, default=str))


# ------------------------------------------------------------- commands

def cmd_validate_env(args) -> int:
    project = EnvironmentProject.load(args.project)
    report = EnvironmentValidator.validate(
        road_def=project.road_def, agent=project.agent, entities=project.entities
    )
    _print_json(report.to_dict())
    return 0 if report.is_valid_for_rl else 1


def cmd_create(args) -> int:
    project = EnvironmentProject.load(args.project)
    scenarios = ScenarioDefinition.get_standard_scenarios()
    scenario = scenarios.get(args.scenario)
    if scenario is None:
        print(f"Unknown scenario '{args.scenario}'. Available: {list(scenarios)}")
        return 1

    report = EnvironmentValidator.validate(
        road_def=project.road_def, agent=project.agent, entities=project.entities
    )
    if not report.is_valid_for_rl and not args.force:
        print("Environment failed validation (use --force to override):")
        _print_json(report.to_dict())
        return 1

    alg_cfg = {}
    if args.alg_config:
        alg_cfg = json.loads(args.alg_config)
    training = TrainingConfig(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        rollout_length=args.rollout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        eval_frequency=args.eval_freq,
        checkpoint_frequency=args.ckpt_freq,
        num_envs=args.num_envs,
        max_wall_seconds=args.max_wall_seconds,
        algorithm_config=alg_cfg,
    )
    evaluation = EvaluationConfig(
        eval_seeds=[int(s) for s in args.eval_seeds.split(",")] if args.eval_seeds else [args.seed],
        num_episodes=args.eval_episodes,
    )
    manifest = ExperimentManifest.from_project(
        project=project, scenario=scenario,
        training=training, evaluation=evaluation,
        name=args.name, random_seed=args.seed,
    )
    mgr = _mgr(args)
    exp_dir = mgr.create(manifest)
    print(f"Created experiment {manifest.experiment_id}")
    print(f"  fingerprint: {manifest.experiment_fingerprint}")
    print(f"  dir:         {exp_dir}")
    return 0


def cmd_list(args) -> int:
    _print_json(_mgr(args).list_experiments())
    return 0


def cmd_show(args) -> int:
    _print_json(_mgr(args).load(args.experiment_id).to_dict())
    return 0


def cmd_launch(args) -> int:
    mgr = _mgr(args)
    manifest = mgr.load(args.experiment_id)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    orch = LocalTrainingOrchestrator(experiments_root=args.root)
    run_id = orch.launch(manifest, exp_dir, trainer=args.trainer, env_mode=args.env_mode)
    mgr.mark_launched(args.experiment_id)
    print(f"Launched run {run_id}")
    if args.wait:
        summary = orch.wait(exp_dir, run_id, timeout_s=args.wait)
        print(f"status: {summary['status']}  timesteps: {summary['current_timestep']}")
    return 0


def cmd_batch(args) -> int:
    mgr = _mgr(args)
    manifest = mgr.load(args.experiment_id)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    specs = expand_run_specs(manifest, seeds=args.seeds, scenario_ids=args.scenarios)
    rmg = RunManager()
    created = []
    for spec in specs:
        run = rmg.create_run(exp_dir, seed=spec["seed"],
                             experiment_id=manifest.experiment_id)
        created.append(run.run_id)
    print(f"Created {len(created)} queued runs:")
    for rid in created:
        print(f"  {rid}")
    return 0


def cmd_trainers(args) -> int:
    _print_json(TRAINER_CAPABILITIES)
    return 0


def cmd_batch_run(args) -> int:
    mgr = _mgr(args)
    manifest = mgr.load(args.experiment_id)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    if getattr(args, "worker", None):
        if not getattr(args, "token", None):
            print("--worker requires --token")
            return 1
        from sim_experiment.remote_worker import RemoteWorkerAdapter
        from sim_experiment.scheduler import Worker
        host, _, port_s = args.worker.rpartition(":")
        adapter = RemoteWorkerAdapter(host, int(port_s), args.token)
        adapter.register()
        scheduler = BatchScheduler(
            experiments_root=args.root,
            workers=[Worker(worker_id=f"remote_{args.worker}",
                            capabilities=adapter.capabilities(),
                            adapter=adapter)])
    else:
        scheduler = BatchScheduler(experiments_root=args.root,
                                   max_workers=args.workers)
    batch = scheduler.create_batch(
        manifest, exp_dir,
        specs=expand_run_specs(manifest, seeds=args.seeds,
                               scenario_ids=args.scenarios),
        trainer=args.trainer, env_mode=args.env_mode,
        retry_policy=RetryPolicy(max_retries=args.max_retries),
    )
    print(f"batch_id: {batch['batch_id']}  jobs: {batch['jobs']}  "
          f"workers: {args.workers}")
    result = scheduler.run_until_complete(batch["batch_id"],
                                          timeout_s=args.timeout)
    summary = scheduler.status(batch["batch_id"])
    print(f"status: {result['status']}  "
          f"completed={summary['completed']} failed={summary['failed']} "
          f"cancelled={summary['cancelled']}  duration={result['duration_s']}s")
    for r in result["runs"]:
        print(f"  job {r['job_id']}: {r['status']} run={r['run_id']} "
              f"attempts={len(r['attempts'])}")
    return 0 if summary["failed"] == 0 and summary["cancelled"] == 0 else 1


def cmd_curriculum(args) -> int:
    rd = _run_dir(args, args.run_id)
    state_path = os.path.join(rd, "curriculum_state.json")
    if not os.path.exists(state_path):
        print("Run has no curriculum (no curriculum_state.json).")
        return 0
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    _print_json(state)
    return 0


def cmd_compare(args) -> int:
    mgr = _mgr(args)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    if args.runs:
        run_dirs = [RunManager().run_dir(exp_dir, r.strip())
                    for r in args.runs.split(",") if r.strip()]
        result = compare_runs(run_dirs, metric=args.metric, scope=args.scope,
                              smooth_window=args.smooth)
    else:
        result = compare_runs(list_run_dirs(exp_dir), metric=args.metric,
                              scope=args.scope, smooth_window=args.smooth)
    _print_json(result)
    return 0


def cmd_dataset_export(args) -> int:
    rd = _run_dir(args, args.run_id)
    reasons = (args.termination_reasons.split(",")
               if args.termination_reasons else None)
    report = export_dataset(
        rd, args.dest,
        env_fingerprint=args.env_fingerprint,
        min_return=args.min_return,
        termination_reasons=reasons,
    )
    _print_json(report)
    return 0


def cmd_worker_serve(args) -> int:
    from sim_experiment.remote_worker import WorkerService
    svc = WorkerService(args.host, args.port,
                        experiments_root=args.worker_root,
                        token=args.token)
    svc.start()
    print(f"Worker service listening on {args.host}:{svc.port} "
          f"(root={os.path.abspath(args.worker_root)})")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
    return 0


def cmd_worker_status(args) -> int:
    from sim_experiment.remote_worker import RemoteWorkerAdapter
    caps = RemoteWorkerAdapter(args.host, args.port, args.token).handshake()
    _print_json(caps)
    return 0


def cmd_runs(args) -> int:
    mgr = _mgr(args)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    _print_json(RunManager().list_runs(exp_dir))
    return 0


def _run_dir(args, run_id: str) -> str:
    mgr = _mgr(args)
    return RunManager().run_dir(mgr.experiment_dir(args.experiment_id), run_id)


def cmd_status(args) -> int:
    mgr = _mgr(args)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    orch = LocalTrainingOrchestrator(experiments_root=args.root)
    while True:
        summary = orch.poll(exp_dir, args.run_id)
        line = (f"{summary['status']:12s} step={summary['current_timestep']:>8} "
                f"eps={summary['episode_count']:>4} "
                f"metrics={json.dumps(summary['latest_metrics'])[:80]}")
        print(line)
        if not args.watch or summary["status"] in RunStatus.TERMINAL:
            break
        time.sleep(2.0)
    return 0


def cmd_cancel(args) -> int:
    mgr = _mgr(args)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    summary = LocalTrainingOrchestrator(experiments_root=args.root).cancel(exp_dir, args.run_id)
    print(f"{args.run_id}: {summary['status']}")
    return 0


def cmd_resume(args) -> int:
    mgr = _mgr(args)
    manifest = mgr.load(args.experiment_id)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    rd = _run_dir(args, args.run_id)
    registry = ArtifactRegistry(rd)
    ckpt = registry.latest_of_kind("checkpoint")
    if ckpt is None:
        print(f"No checkpoint found in run {args.run_id}; cannot resume.")
        return 1
    ckpt_abs = os.path.join(rd, ckpt["path"])
    orch = LocalTrainingOrchestrator(experiments_root=args.root)
    new_run_id = orch.launch(
        manifest, exp_dir, trainer=args.trainer, env_mode="inprocess",
        resume_from={"parent_run_id": args.run_id, "checkpoint": ckpt_abs},
    )
    print(f"Resumed as new run {new_run_id} from {ckpt['path']}")
    return 0


def cmd_evaluate(args) -> int:
    mgr = _mgr(args)
    manifest = mgr.load(args.experiment_id)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    rd = _run_dir(args, args.run_id)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt = ArtifactRegistry(rd).latest_of_kind("checkpoint")
        if ckpt is None:
            print("No checkpoint found; pass --checkpoint explicitly.")
            return 1
        ckpt_path = os.path.join(rd, ckpt["path"])
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    env = build_env_from_dicts(manifest.environment, manifest.scenario_configuration,
                               seed=manifest.random_seed)
    policy = make_policy_from_checkpoint(
        ckpt_path, algorithm=manifest.training.algorithm,
        deterministic=manifest.evaluation.deterministic_policy,
    )
    result = evaluate_policy(
        env, policy,
        seeds=manifest.evaluation.eval_seeds,
        num_episodes=manifest.evaluation.num_episodes,
        deterministic=manifest.evaluation.deterministic_policy,
        result_kwargs={
            "checkpoint_path": os.path.relpath(ckpt_path, rd),
            "algorithm": manifest.training.algorithm,
            "env_fingerprint": manifest.environment_fingerprint,
            "scenario_id": manifest.scenario_id,
        },
    )
    eval_dir = os.path.join(rd, "evaluation")
    eval_path = os.path.join(eval_dir, f"{result.eval_id}.json")
    result.save(eval_path)
    ArtifactRegistry(rd).register("evaluation", os.path.relpath(eval_path, rd))
    RunManager().add_artifact_ref(exp_dir, args.run_id, "evaluation",
                                  os.path.relpath(eval_path, rd))
    _print_json(result.aggregate)
    return 0


def cmd_trajectories(args) -> int:
    rd = _run_dir(args, args.run_id)
    episodes = list_episodes(rd)
    print(f"{len(episodes)} trajectory episodes in {os.path.join(rd, 'trajectories')}")
    for e in episodes:
        print(f"  {e['episode_id']}: steps={e['steps']} "
              f"return={e['total_return']} reason={e['termination_reason']}")
    if args.json:
        _print_json(episodes)
    return 0


def cmd_reproduce(args) -> int:
    mgr = _mgr(args)
    report = check_reproducibility(mgr.experiment_dir(args.experiment_id))
    _print_json(report)
    return 0 if report["reproducible"] else 1


def cmd_export(args) -> int:
    dest = _mgr(args).export(args.experiment_id, args.dest)
    print(f"Exported to {dest}")
    return 0


def cmd_archive(args) -> int:
    _mgr(args).archive(args.experiment_id)
    print(f"Archived {args.experiment_id}")
    return 0


def cmd_learn_bench(args) -> int:
    from benchmarks.phase6.benchmark_runner import run_benchmark
    from sim_experiment.convergence import convergence_report
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    results = run_benchmark(config, work_dir=args.work_dir)
    report = convergence_report(results)
    _print_json(report)
    print(f"Results: {results['work_dir']}/benchmark_results.json")
    return 0 if report.get("verdict") != "regressed" else 1


def cmd_dataset_validate(args) -> int:
    from sim_experiment.dataset import validate_dataset
    rep = validate_dataset(args.dataset_dir)
    _print_json(rep)
    return 0 if rep["valid"] else 1


def cmd_dataset_split(args) -> int:
    from sim_experiment.dataset import split_dataset
    ratios = {"train": 1.0 - args.val_frac - args.test_frac,
              "val": args.val_frac, "test": args.test_frac}
    splits = split_dataset(args.dataset_dir, seed=args.seed, ratios=ratios)
    _print_json({k: len(v) for k, v in splits.items()})
    return 0


def cmd_dataset_stats(args) -> int:
    from sim_experiment.dataset_inspect import inspect_dataset
    _print_json(inspect_dataset(args.dataset_dir))
    return 0


def cmd_train_bc(args) -> int:
    mgr = _mgr(args)
    manifest = mgr.load(args.experiment_id)
    exp_dir = mgr.experiment_dir(args.experiment_id)
    alg_cfg = {"bc_epochs": args.epochs, "val_ratio": args.val_frac}
    if manifest.training.algorithm != "bc":
        # Override the algorithm to bc for this run; the dataset carries the
        # learned behavior, the experiment pins env/scenario fingerprints.
        manifest.training.algorithm = "bc"
    orch = LocalTrainingOrchestrator(experiments_root=args.root)
    run_id = orch.launch(
        manifest, exp_dir, trainer="bc",
        run_overrides={"bc_dataset_dir": args.dataset,
                       "seed": args.seed,
                       "algorithm_config": alg_cfg})
    mgr.mark_launched(args.experiment_id)
    print(f"Launched BC run {run_id} on dataset {args.dataset}")
    if args.wait:
        summary = orch.wait(exp_dir, run_id, timeout_s=args.wait)
        print(f"status: {summary['status']}")
        return 0 if summary["status"] == "COMPLETED" else 1
    return 0


def cmd_worker_register(args) -> int:
    from sim_experiment.remote_worker import RemoteWorkerAdapter
    resp = RemoteWorkerAdapter(args.host, args.port, args.token).register()
    _print_json(resp)
    return 0


def cmd_worker_list(args) -> int:
    from sim_experiment.remote_worker import RemoteWorkerAdapter
    resp = RemoteWorkerAdapter(args.host, args.port, args.token).handshake()
    _print_json(resp.get("capabilities", {}))
    return 0


def cmd_benchmark(args) -> int:
    import numpy as np
    from sim_env.templates import EnvironmentTemplateManager

    project = EnvironmentTemplateManager.create_project_from_template(
        args.template or "lane_following"
    )
    env_dict = project.to_dict()
    scen_dict = project.scenario_def.to_dict()

    results = []
    for n in [int(x) for x in args.envs.split(",")]:
        pool = HeadlessEnvPool(env_dict, scen_dict, num_envs=n, base_seed=1000)
        pool.reset_all()
        t0 = time.perf_counter()
        steps_done = 0
        while steps_done < args.steps:
            for env in pool.envs:
                _, _, term, trunc, _ = env.step([0.0, 0.5, 0.0])
                steps_done += 1
                if term or trunc:
                    env.reset()
        elapsed = time.perf_counter() - t0
        sps = steps_done / elapsed
        results.append({
            "num_envs": n,
            "steps": steps_done,
            "elapsed_s": round(elapsed, 3),
            "env_steps_per_sec": round(sps, 1),
            "per_env_sps": round(sps / n, 1),
        })
        print(f"envs={n:2d}  steps={steps_done:6d}  {elapsed:6.2f}s  "
              f"{sps:8.1f} env-steps/s  ({sps/n:8.1f}/env)")
    _print_json(results)
    return 0


# ------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sim_experiment.cli",
                                description="Training & Experiment Platform CLI")
    p.add_argument("--root", default="experiments",
                   help="Experiments root directory (default: experiments/)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("validate-env")
    s.add_argument("--project", required=True)
    s.set_defaults(fn=cmd_validate_env)

    s = sub.add_parser("create")
    s.add_argument("--project", required=True)
    s.add_argument("--scenario", default="basic_lane_following")
    s.add_argument("--name", default="experiment")
    s.add_argument("--algorithm", default="ppo")
    s.add_argument("--timesteps", type=int, default=50000)
    s.add_argument("--rollout", type=int, default=1024)
    s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--gamma", type=float, default=0.99)
    s.add_argument("--gae-lambda", type=float, default=0.95)
    s.add_argument("--eval-freq", type=int, default=0)
    s.add_argument("--ckpt-freq", type=int, default=10000)
    s.add_argument("--num-envs", type=int, default=1)
    s.add_argument("--max-wall-seconds", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--eval-seeds", default="")
    s.add_argument("--eval-episodes", type=int, default=5)
    s.add_argument("--alg-config", default="")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_create)

    s = sub.add_parser("list")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("show")
    s.add_argument("experiment_id")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("launch")
    s.add_argument("experiment_id")
    s.add_argument("--trainer", default="ppo")
    s.add_argument("--env-mode", default="inprocess",
                   choices=["inprocess", "process", "tcp", "tcp_multi"])
    s.add_argument("--wait", type=float, default=0, help="Wait up to N seconds for completion")
    s.set_defaults(fn=cmd_launch)

    s = sub.add_parser("batch")
    s.add_argument("experiment_id")
    s.add_argument("--seeds", type=int, nargs="*", default=None)
    s.add_argument("--scenarios", nargs="*", default=None)
    s.set_defaults(fn=cmd_batch)

    s = sub.add_parser("batch-run")
    s.add_argument("experiment_id")
    s.add_argument("--seeds", type=int, nargs="*", default=None)
    s.add_argument("--scenarios", nargs="*", default=None)
    s.add_argument("--trainer", default="ppo")
    s.add_argument("--env-mode", default="inprocess",
                   choices=["inprocess", "process", "tcp", "tcp_multi"])
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--worker", default=None,
                   help="remote worker host:port (uses --token)")
    s.add_argument("--token", default=None,
                   help="shared token for --worker / worker commands")
    s.add_argument("--max-retries", type=int, default=0)
    s.add_argument("--timeout", type=float, default=600.0)
    s.set_defaults(fn=cmd_batch_run)

    s = sub.add_parser("trainers")
    s.set_defaults(fn=cmd_trainers)

    s = sub.add_parser("curriculum")
    s.add_argument("experiment_id")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_curriculum)

    s = sub.add_parser("compare")
    s.add_argument("experiment_id")
    s.add_argument("--metric", default="reward")
    s.add_argument("--scope", default="episode")
    s.add_argument("--smooth", type=int, default=10)
    s.add_argument("--runs", default="", help="comma-separated run ids (default: all)")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("dataset-export")
    s.add_argument("experiment_id")
    s.add_argument("run_id")
    s.add_argument("--dest", required=True)
    s.add_argument("--min-return", type=float, default=None)
    s.add_argument("--env-fingerprint", default=None)
    s.add_argument("--termination-reasons", default=None,
                   help="comma-separated reasons to include")
    s.set_defaults(fn=cmd_dataset_export)

    s = sub.add_parser("worker-serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=9100)
    s.add_argument("--token", required=True)
    s.add_argument("--worker-root", default="experiments")
    s.set_defaults(fn=cmd_worker_serve)

    s = sub.add_parser("worker-status")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--token", required=True)
    s.set_defaults(fn=cmd_worker_status)

    s = sub.add_parser("runs")
    s.add_argument("experiment_id")
    s.set_defaults(fn=cmd_runs)

    s = sub.add_parser("status")
    s.add_argument("experiment_id")
    s.add_argument("run_id")
    s.add_argument("--watch", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("cancel")
    s.add_argument("experiment_id")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_cancel)

    s = sub.add_parser("resume")
    s.add_argument("experiment_id")
    s.add_argument("run_id")
    s.add_argument("--trainer", default="ppo")
    s.set_defaults(fn=cmd_resume)

    s = sub.add_parser("evaluate")
    s.add_argument("experiment_id")
    s.add_argument("run_id")
    s.add_argument("--checkpoint", default=None)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("trajectories")
    s.add_argument("experiment_id")
    s.add_argument("run_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_trajectories)

    s = sub.add_parser("reproduce")
    s.add_argument("experiment_id")
    s.set_defaults(fn=cmd_reproduce)

    s = sub.add_parser("export")
    s.add_argument("experiment_id")
    s.add_argument("--dest", required=True)
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("archive")
    s.add_argument("experiment_id")
    s.set_defaults(fn=cmd_archive)

    s = sub.add_parser("benchmark")
    s.add_argument("--envs", default="1,2,4")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--template", default=None)
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("learn-bench")
    s.add_argument("--config", required=True)
    s.add_argument("--work-dir", default=None)
    s.set_defaults(fn=cmd_learn_bench)

    s = sub.add_parser("dataset-validate")
    s.add_argument("dataset_dir")
    s.set_defaults(fn=cmd_dataset_validate)

    s = sub.add_parser("dataset-split")
    s.add_argument("dataset_dir")
    s.add_argument("--val-frac", type=float, default=0.1)
    s.add_argument("--test-frac", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_dataset_split)

    s = sub.add_parser("dataset-stats")
    s.add_argument("dataset_dir")
    s.set_defaults(fn=cmd_dataset_stats)

    s = sub.add_parser("train-bc")
    s.add_argument("experiment_id")
    s.add_argument("--dataset", required=True,
                   help="transitions_v1 dataset directory")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--val-frac", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--wait", type=float, default=0)
    s.set_defaults(fn=cmd_train_bc)

    s = sub.add_parser("worker-register")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--token", required=True)
    s.set_defaults(fn=cmd_worker_register)

    s = sub.add_parser("worker-list")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--token", required=True)
    s.set_defaults(fn=cmd_worker_list)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
