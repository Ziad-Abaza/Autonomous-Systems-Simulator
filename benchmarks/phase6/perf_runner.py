"""
Phase 6 performance measurement — env_mode scaling, measured not assumed.

Compares the four execution backends stepping the SAME env family:

    inprocess  — N SimulationEnvironments in the trainer process
    process    — ProcessVectorEnv: N spawn'd child processes, pipe IPC
    tcp        — N headless sim processes (one server each), N TCP clients
    tcp_multi  — ONE headless process hosting N envs (SimServerMulti)

Usage:
    python benchmarks/phase6/perf_runner.py [--envs 1,2,4] [--steps 400]
        [--modes inprocess,process] [--out docs/PHASE_6_PERFORMANCE_RAW.json]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ACTION = [0.0, 0.5, 0.0]


def _project_env_args():
    from sim_env.templates import EnvironmentTemplateManager
    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    return project


def _total_steps(envs_or_vec, steps_per_env, num_envs, is_vec):
    """Steps every env `steps_per_env` times; returns (elapsed, steps_done)."""
    t0 = time.perf_counter()
    done = 0
    if is_vec:
        for _ in range(steps_per_env):
            envs_or_vec.step_all([ACTION] * num_envs)
            done += num_envs
    else:
        for _ in range(steps_per_env):
            for env in envs_or_vec:
                _, _, term, trunc, _ = env.step(ACTION)
                done += 1
                if term or trunc:
                    env.reset()
    return time.perf_counter() - t0, done


def bench_inprocess(num_envs: int, steps_per_env: int) -> dict:
    from sim_experiment.headless import build_env_from_dicts
    project = _project_env_args()
    env_dict = project.to_dict()
    scen_dict = project.scenario_def.to_dict()
    envs = [build_env_from_dicts(env_dict, scen_dict, seed=1000 + i)
            for i in range(num_envs)]
    for e in envs:
        e.reset()
    elapsed, done = _total_steps(envs, steps_per_env, num_envs, is_vec=False)
    return {"mode": "inprocess", "num_envs": num_envs, "steps": done,
            "elapsed_s": round(elapsed, 3),
            "steps_per_sec": round(done / elapsed, 1),
            "per_env_sps": round(done / elapsed / num_envs, 1)}


def bench_process(num_envs: int, steps_per_env: int) -> dict:
    from sim_experiment.process_env import ProcessVectorEnv
    project = _project_env_args()
    env_dict = project.to_dict()
    scen_dict = project.scenario_def.to_dict()

    t_setup = time.perf_counter()
    vec = ProcessVectorEnv(env_dict=env_dict, scenario_dict=scen_dict,
                           seeds=[1000 + i for i in range(num_envs)])
    setup_s = time.perf_counter() - t_setup
    try:
        vec.reset_all([1000 + i for i in range(num_envs)])
        elapsed, done = _total_steps(vec, steps_per_env, num_envs, is_vec=True)
    finally:
        vec.close()
    return {"mode": "process", "num_envs": num_envs, "steps": done,
            "setup_s": round(setup_s, 2),
            "elapsed_s": round(elapsed, 3),
            "steps_per_sec": round(done / elapsed, 1),
            "per_env_sps": round(done / elapsed / num_envs, 1)}


def bench_tcp(num_envs: int, steps_per_env: int, shared: bool) -> dict:
    from sim_experiment.headless import HeadlessSimProcessPool
    from sim_client.gym_env import SimGymEnv

    mode = "tcp_multi" if shared else "tcp"
    t_setup = time.perf_counter()
    pool = HeadlessSimProcessPool(num_envs, shared_process=shared)
    ports = pool.start()
    clients = [SimGymEnv(host="127.0.0.1", port=p) for p in ports]
    setup_s = time.perf_counter() - t_setup
    try:
        for c in clients:
            c.reset(seed=1000)
        elapsed, done = _total_steps(clients, steps_per_env, num_envs,
                                     is_vec=False)
    finally:
        for c in clients:
            c.close()
        pool.stop()
    return {"mode": mode, "num_envs": num_envs, "steps": done,
            "setup_s": round(setup_s, 2), "procs": len(pool.procs),
            "elapsed_s": round(elapsed, 3),
            "steps_per_sec": round(done / elapsed, 1),
            "per_env_sps": round(done / elapsed / num_envs, 1)}


BENCH = {
    "inprocess": lambda n, s: bench_inprocess(n, s),
    "process": lambda n, s: bench_process(n, s),
    "tcp": lambda n, s: bench_tcp(n, s, shared=False),
    "tcp_multi": lambda n, s: bench_tcp(n, s, shared=True),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default="1,2,4")
    parser.add_argument("--steps", type=int, default=400,
                        help="steps per env per measurement")
    parser.add_argument("--modes", default="inprocess,process",
                        help="comma list: inprocess,process,tcp,tcp_multi")
    parser.add_argument("--out", default="docs/PHASE_6_PERFORMANCE_RAW.json")
    args = parser.parse_args()

    counts = [int(x) for x in args.envs.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    report = {"benchmarked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "steps_per_env": args.steps, "results": []}
    for mode in modes:
        for n in counts:
            print(f"[bench] {mode} num_envs={n} ...", flush=True)
            r = BENCH[mode](n, args.steps)
            report["results"].append(r)
            print(f"        {r['steps_per_sec']} steps/s "
                  f"({r['per_env_sps']}/env)  elapsed={r['elapsed_s']}s",
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
