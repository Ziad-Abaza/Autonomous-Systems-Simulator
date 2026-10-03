---
noteId: "e0a416d0bf1311f1a29f1fbaabbd87c8"
tags: []

---

# Phase 6 Performance

Measured with `benchmarks/phase6/perf_runner.py` (300 steps/env,
lane_following env, this machine). Raw data: `PHASE_6_PERFORMANCE_RAW.json`.

## Steps/second by env_mode × num_envs

| mode | 1 env | 2 envs | 4 envs |
|---|---|---|---|
| inprocess | 392.5 | 386.9 | 366.8 |
| process | 361.3 | 629.2 | 1056.9 |
| tcp (1 sim/env) | 125.2 | 180.2 | 179.9 |
| tcp_multi (1 sim, N envs) | 124.2 | 325.2 | 315.4 |

## Per-env steps/second

| mode | 1 | 2 | 4 |
|---|---|---|---|
| inprocess | 392.5 | 193.5 | 91.7 |
| process | 361.3 | 314.6 | 264.2 |
| tcp | 125.2 | 90.1 | 45.0 |
| tcp_multi | 124.2 | 162.6 | 78.8 |

## Reading

- **inprocess** is GIL-bound: aggregate ~flat (392→367), per-env halves as
  envs double. Cheapest transport, no isolation.
- **process** is the scaling winner: aggregate +63% (1→2) and +68% (2→4)
  — the Pipe round-trip is amortized by stepping all workers per cycle.
  Best local throughput with real isolation.
- **tcp** is latency-bound (per-step socket round-trip); aggregate barely
  scales with sequential clients. Its value is the real serialization
  boundary + LAN distribution, not local speed.
- **tcp_multi** outperforms tcp at the same client count (one server loop
  amortizes accept/poll) — good middle ground for many envs in one
  simulator process.

## Reproduce

```powershell
python benchmarks/phase6/perf_runner.py --envs 1,2,4 --steps 300 `
    --modes inprocess,process,tcp,tcp_multi --out docs/PHASE_6_PERFORMANCE_RAW.json
```

Numbers are machine-dependent; rerun on the target hardware before
drawing deployment conclusions.
