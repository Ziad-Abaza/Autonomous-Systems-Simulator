"""
Phase 5 performance benchmarks — measured, not assumed.

Measures:
  1. In-process env throughput at num_envs = 1, 2, 4 (steps/sec)
  2. PPO trainer end-to-end SPS at rollout sizes (real subprocess run)
  3. Batch scheduler dispatch overhead (create_batch + tick latency)
  4. Curriculum controller overhead per evaluation decision

Usage: python benchmarks/phase5_benchmarks.py [--out docs/PHASE_5_PERFORMANCE_RAW.json]
"""

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def bench_env_throughput(counts=(1, 2, 4), steps_per_env=600):
    from sim_env.templates import EnvironmentTemplateManager
    from sim_experiment.headless import HeadlessEnvPool

    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    env_dict = project.to_dict()
    scen_dict = project.scenario_def.to_dict()

    results = []
    for n in counts:
        pool = HeadlessEnvPool(env_dict, scen_dict, num_envs=n, base_seed=1000)
        pool.reset_all()
        t0 = time.perf_counter()
        done = 0
        while done < steps_per_env * n:
            for env in pool.envs:
                _, _, term, trunc, _ = env.step([0.0, 0.5, 0.0])
                done += 1
                if term or trunc:
                    env.reset()
        elapsed = time.perf_counter() - t0
        results.append({
            "num_envs": n,
            "total_steps": done,
            "elapsed_s": round(elapsed, 3),
            "steps_per_sec": round(done / elapsed, 1),
            "per_env_sps": round(done / elapsed / n, 1),
        })
    return results


def bench_ppo_sps(timesteps=512, rollout=256):
    """One real PPO run through the orchestrator; report wall-clock SPS."""
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    from sim_experiment.manager import ExperimentManager
    from sim_experiment.orchestrator import LocalTrainingOrchestrator
    from sim_env.templates import EnvironmentTemplateManager

    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    with tempfile.TemporaryDirectory() as td:
        mgr = ExperimentManager(root_dir=os.path.join(td, "experiments"))
        m = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=TrainingConfig(
                algorithm="ppo", total_timesteps=timesteps,
                rollout_length=rollout, batch_size=rollout, epochs=2,
                eval_frequency=0, checkpoint_frequency=0, num_envs=1),
            evaluation=EvaluationConfig(), name="bench_ppo", random_seed=7)
        exp_dir = mgr.create(m)
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        rid = orch.launch(m, exp_dir, trainer="ppo")
        summary = orch.wait(exp_dir, rid, timeout_s=600)
        result_path = os.path.join(exp_dir, "runs", rid, "run_result.json")
        result = {}
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
        return {
            "status": summary["status"],
            "timesteps": result.get("total_timesteps", timesteps),
            "sps": result.get("sps"),
            "wall_clock_time": result.get("wall_clock_time"),
        }


def bench_scheduler_dispatch(jobs=8):
    """create_batch + tick dispatch latency with dummy trainer."""
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    from sim_experiment.manager import ExperimentManager
    from sim_experiment.scheduler import BatchScheduler
    from sim_env.templates import EnvironmentTemplateManager

    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    with tempfile.TemporaryDirectory() as td:
        mgr = ExperimentManager(root_dir=os.path.join(td, "experiments"))
        m = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=TrainingConfig(algorithm="dummy", total_timesteps=10),
            evaluation=EvaluationConfig(), name="bench_batch", random_seed=1)
        exp_dir = mgr.create(m)
        s = BatchScheduler(mgr.root_dir, max_workers=2)
        t0 = time.perf_counter()
        batch = s.create_batch(m, exp_dir,
                               specs=[{"seed": i} for i in range(jobs)],
                               trainer="dummy")
        create_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        s.tick()
        tick_ms = (time.perf_counter() - t0) * 1000
        result = s.run_until_complete(batch["batch_id"], timeout_s=300)
        return {
            "jobs": jobs,
            "create_batch_ms": round(create_ms, 2),
            "first_tick_ms": round(tick_ms, 2),
            "batch_wall_s": result["duration_s"],
            "status": result["status"],
        }


def bench_curriculum_eval(evaluations=1000):
    """CurriculumController.evaluate_advancement cost per call."""
    from sim_experiment.curriculum_runtime import CurriculumController
    from sim_env.curriculum import CurriculumDefinition

    c = CurriculumController(CurriculumDefinition.create_default(), base_seed=1)
    agg = {"mean_reward": 0.5, "collision_rate": 0.1, "completion_rate": 0.7}
    t0 = time.perf_counter()
    for _ in range(evaluations):
        c.evaluate_advancement(agg, timestep=1)
    elapsed = time.perf_counter() - t0
    return {
        "decisions": evaluations,
        "elapsed_s": round(elapsed, 4),
        "us_per_decision": round(elapsed / evaluations * 1e6, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/PHASE_5_PERFORMANCE_RAW.json")
    parser.add_argument("--skip-ppo", action="store_true")
    args = parser.parse_args()

    report = {"benchmarked_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("[1/4] env throughput ...")
    report["env_throughput"] = bench_env_throughput()
    for r in report["env_throughput"]:
        print(f"   envs={r['num_envs']}: {r['steps_per_sec']} steps/s "
              f"({r['per_env_sps']}/env)")

    if args.skip_ppo:
        report["ppo"] = {"skipped": True}
    else:
        print("[2/4] PPO end-to-end SPS (real subprocess run) ...")
        report["ppo"] = bench_ppo_sps()
        print(f"   {report['ppo']}")

    print("[3/4] batch scheduler dispatch ...")
    report["scheduler"] = bench_scheduler_dispatch()
    print(f"   {report['scheduler']}")

    print("[4/4] curriculum decision cost ...")
    report["curriculum"] = bench_curriculum_eval()
    print(f"   {report['curriculum']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
