---
noteId: "42ee8870bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Parallel Environments

Two parallelism paths, both external to the simulation core.

## In-process (`env_mode="inprocess"`, `num_envs > 1`)

- The trainer builds N independent `SimulationEnvironment` instances
  (`seed + i`), all stepping in the trainer process.
- `PPORunner` accepts a list of envs: each env contributes a **contiguous
  rollout segment** (`num_steps // num_envs` per env) and GAE is computed
  per segment — per-env bootstrapping stays correct.
- Cheap: no sockets, no subprocesses. Shared interpreter means linear
  scaling is limited by CPU (Python GIL + physics cost).

## TCP (`env_mode="tcp"`, `num_envs > 1`)

- Orchestrator spawns N `main.py --headless --port` simulator processes;
  contract carries `tcp.ports`.
- Trainer connects via `SimGymEnv` per port — true process isolation;
  each simulator owns its RNG, clock, and physics state.
- NDJSON protocol 2.0 with version negotiation.

## Independent seeds

- Env seeds: `contract.seed + env_index`.
- Trainer seeds torch/numpy with `contract.seed`; env resets use
  `runner.seed + env_idx`.
- Two envs with different seeds produce independent episode sequences
  (verified by test: `test_independent_envs_and_seeds`).

## Measured throughput (this machine, see PHASE_4_PERFORMANCE)

| Envs (in-process) | env-steps/s | per-env sps |
|---|---|---|
| 1 | 352 | 352 |
| 2 | 340 | 170 |
| 4 | 332 | 83 |

Physics+sensor cost dominates; in-process parallelism gives near-linear
*episode diversity* but CPU-bound aggregate throughput. For wall-clock
scaling, run multiple trainer processes or the TCP mode on separate
machines/cores.
