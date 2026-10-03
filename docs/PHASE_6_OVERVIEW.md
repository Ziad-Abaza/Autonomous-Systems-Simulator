---
noteId: "c6513290bf1311f1a29f1fbaabbd87c8"
tags: []

---

# Phase 6 Overview

Phase 6 turns the Phase 5 experiment runner into a scalable, verifiable
training platform. Goal: correctness → reproducibility → measurable
performance → clean architecture → security → maintainability → features.

## What shipped

| Area | Deliverable |
|---|---|
| Parallel envs | `SyncVectorEnv`, `ProcessVectorEnv` (spawn+pipe, Windows-safe), runner integration for PPO/SAC/DQN (`env_mode="inprocess"|"process"`) |
| TCP envs | `SimGymEnv` × N wrapped in `SyncVectorEnv`; `SimServerMulti` + `env_mode="tcp_multi"` (one process, N envs, one port) |
| Scenario control | `SimulationEnvironment.set_scenario`, scenario-entity cleanup + broadphase consistency, protocol 2.1 `SET_SCENARIO` — curriculum works over TCP |
| Learning validation | `benchmarks/phase6/` baseline→train→eval→resume→eval runner + `sim_experiment/convergence.py` verdicts (improved / no_improvement / regressed / unstable) |
| Datasets | `transitions_v1` validation, deterministic splits, stats/inspect, seed-keyed trajectory sampling, header provenance (`episode_seed`, `env_index`, `curriculum_stage_index`), eval transition export |
| Imitation learning | `sim_experiment/bc/` + `bc_trainer` — dataset validation → split → MLP train → checkpoint → frozen-policy eval through the standard contract |
| Worker hardening | worker_id registration, persistent `WorkerRegistry`, heartbeat leases, OFFLINE + job reclaim, duplicate-dispatch guard, root checks on LAUNCH/POLL/CANCEL, per-policy retry types |
| UI | WORKERS section, dataset preview + episode browser, multi-series comparison chart |
| CLI | `learn-bench`, `dataset-validate/-split/-stats`, `train-bc`, `worker-register/-list`, `batch-run --worker/--token`, `--env-mode process/tcp_multi` |

## Architecture map

```
ExperimentManifest ──> LocalTrainingOrchestrator ──> contract.json
        │                                                   │
        │                      env_mode                      ▼
        │         ┌──────────┬──────────┬──────────┬──────────────┐
        │      inprocess  process    tcp      tcp_multi       trainer
        │      (SyncVec) (ProcVec) (N sims) (1 multi sim)   subprocess
        │                                                   │
        ▼                                                   ▼
   BatchScheduler ── workers ──> metrics.jsonl / checkpoints /
        │                        trajectories / eval artifacts
        ▼                                                   │
   WorkerRegistry + heartbeats                              ▼
                                        datasets → validate/split → BC
```

## Verifying

```powershell
python -m pytest tests -q                                  # full suite
python benchmarks/phase6/benchmark_runner.py --config benchmarks/phase6/benchmark_config.json
python benchmarks/phase6/perf_runner.py --envs 1,2,4 --modes inprocess,process,tcp,tcp_multi
```

See `PHASE_6_FINAL_REPORT.md` for per-claim verification status and
`PHASE_6_PERFORMANCE.md` for measured backend throughput.
