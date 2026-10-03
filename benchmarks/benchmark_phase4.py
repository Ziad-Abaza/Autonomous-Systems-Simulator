"""
Phase 4 throughput benchmark.

Measures, reproducibly:
- in-process env.step throughput for 1/2/4 environments
- sensor configuration impact (full suite vs lidar-only vs state-only)
- metrics pipeline overhead (buffered MetricsWriter vs none)
- orchestration overhead (contract build + subprocess launch latency)

Writes results to benchmarks/phase4_results.json.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sim_env.templates import EnvironmentTemplateManager
from sim_experiment.headless import build_env_from_dicts, HeadlessEnvPool
from sim_experiment.metrics import MetricsWriter


def _load_project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


def bench_env_throughput(num_envs: int, steps: int = 3000) -> dict:
    proj = _load_project()
    pool = HeadlessEnvPool(proj.to_dict(), proj.scenario_def.to_dict(),
                           num_envs=num_envs, base_seed=1000)
    pool.reset_all()
    action = [0.0, 0.5, 0.0]
    t0 = time.perf_counter()
    done = 0
    while done < steps:
        for env in pool.envs:
            _, _, term, trunc, _ = env.step(action)
            done += 1
            if term or trunc:
                env.reset()
    elapsed = time.perf_counter() - t0
    return {
        "num_envs": num_envs,
        "steps": done,
        "elapsed_s": round(elapsed, 3),
        "env_steps_per_sec": round(done / elapsed, 1),
        "per_env_sps": round(done / elapsed / num_envs, 1),
        "avg_step_ms": round(elapsed / done * 1000, 3),
    }


def bench_sensor_configs(steps: int = 1500) -> list:
    proj = _load_project()
    results = []
    for label, builder in (
        ("full_suite", None),
        ("lidar_only", "lidar"),
        ("state_only", "state"),
    ):
        if builder is None:
            env = build_env_from_dicts(proj.to_dict(), proj.scenario_def.to_dict(), seed=7)
        elif builder == "lidar":
            from sim_core.sensors.sensor_manager import SensorManager
            from sim_core.sensors.vehicle_state_sensor import VehicleStateSensor
            from sim_core.sensors.raycast_sensor import RaycastSensor
            sm = SensorManager()
            sm.add_sensor(VehicleStateSensor(name="vehicle_state", update_frequency_hz=60.0))
            sm.add_sensor(RaycastSensor(name="lidar_rays", num_rays=15, fov_degrees=180.0,
                                       max_range=40.0, update_frequency_hz=30.0))
            env = build_env_from_dicts(proj.to_dict(), proj.scenario_def.to_dict(), seed=7)
            env.sensors = sm
        else:
            from sim_core.sensors.sensor_manager import SensorManager
            from sim_core.sensors.vehicle_state_sensor import VehicleStateSensor
            sm = SensorManager()
            sm.add_sensor(VehicleStateSensor(name="vehicle_state", update_frequency_hz=60.0))
            env = build_env_from_dicts(proj.to_dict(), proj.scenario_def.to_dict(), seed=7)
            env.sensors = sm

        env.reset(seed=7)
        action = [0.0, 0.5, 0.0]
        t0 = time.perf_counter()
        for _ in range(steps):
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                env.reset()
        elapsed = time.perf_counter() - t0
        results.append({
            "config": label,
            "steps": steps,
            "env_steps_per_sec": round(steps / elapsed, 1),
            "avg_step_ms": round(elapsed / steps * 1000, 3),
        })
    return results


def bench_metrics_overhead(steps: int = 1500) -> dict:
    proj = _load_project()
    env = build_env_from_dicts(proj.to_dict(), proj.scenario_def.to_dict(), seed=5)
    env.reset(seed=5)
    action = [0.0, 0.5, 0.0]

    t0 = time.perf_counter()
    for i in range(steps):
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset()
    no_metrics = time.perf_counter() - t0

    import tempfile
    mpath = os.path.join(tempfile.mkdtemp(), "metrics.jsonl")
    writer = MetricsWriter(mpath, buffer_size=64)
    env.reset(seed=5)
    t0 = time.perf_counter()
    for i in range(steps):
        _, _, term, trunc, info = env.step(action)
        # realistic cadence: step-scope metrics are buffered, not per-step disk I/O
        writer.write("step", timestep=i, metrics={"reward": 0.1, "speed": info.get("speed", 0.0)})
        if term or trunc:
            env.reset()
    writer.close()
    with_metrics = time.perf_counter() - t0

    return {
        "steps": steps,
        "no_metrics_sps": round(steps / no_metrics, 1),
        "with_metrics_sps": round(steps / with_metrics, 1),
        "overhead_pct": round((with_metrics - no_metrics) / no_metrics * 100, 2),
    }


def bench_orchestration_overhead() -> dict:
    """Contract build time (launch-side CPU work)."""
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    from sim_experiment.trainer_contract import build_contract, validate_contract
    import tempfile

    proj = _load_project()
    manifest = ExperimentManifest.from_project(
        project=proj, scenario=proj.scenario_def,
        training=TrainingConfig(algorithm="ppo", total_timesteps=100),
        evaluation=EvaluationConfig(), name="bench", random_seed=1,
    )
    t0 = time.perf_counter()
    for _ in range(20):
        c = build_contract(manifest, tempfile.gettempdir(), tempfile.gettempdir(), "r")
        validate_contract(c)
    elapsed = (time.perf_counter() - t0) / 20
    return {"contract_build_ms": round(elapsed * 1000, 2)}


def main():
    print("=== Phase 4 Throughput Benchmark ===")
    results = {
        "machine": {"platform": sys.platform, "python": sys.version.split()[0]},
        "env_throughput": [],
        "sensor_configs": [],
        "metrics_overhead": {},
        "orchestration": {},
    }

    for n in (1, 2, 4):
        r = bench_env_throughput(n)
        results["env_throughput"].append(r)
        print(f"envs={n}: {r['env_steps_per_sec']} env-steps/s ({r['per_env_sps']}/env, {r['avg_step_ms']} ms/step)")

    for r in bench_sensor_configs():
        results["sensor_configs"].append(r)
        print(f"sensors={r['config']}: {r['env_steps_per_sec']} steps/s ({r['avg_step_ms']} ms/step)")

    r = bench_metrics_overhead()
    results["metrics_overhead"] = r
    print(f"metrics overhead: {r['overhead_pct']}% ({r['with_metrics_sps']} vs {r['no_metrics_sps']} sps)")

    r = bench_orchestration_overhead()
    results["orchestration"] = r
    print(f"contract build+validate: {r['contract_build_ms']} ms")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase4_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
