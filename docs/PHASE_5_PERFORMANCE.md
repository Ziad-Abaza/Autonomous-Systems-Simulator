---
noteId: "f4e5d570bf0911f1a29f1fbaabbd87c8"
tags: []

---

# Phase 5 Performance Benchmarks

All numbers are **measured** on the development machine
(Windows, Python 3.13, CPU-only torch). Raw data:
`docs/PHASE_5_PERFORMANCE_RAW.json`. Reproduce with:

```
python benchmarks/phase5_benchmarks.py
```

## 1. Environment throughput (in-process `HeadlessEnvPool`)

| num_envs | total steps/s | steps/s per env |
|---|---|---|
| 1 | 394.1 | 394.1 |
| 2 | 393.5 | 196.7 |
| 4 | 363.1 | 90.8 |

Read: in-process envs are stepped sequentially inside one process — aggregate
throughput is roughly constant; each env's effective SPS drops with count.
Sequential multi-env rollout does **not** scale linearly; true parallelism
requires subprocess workers (Phase 5 scheduler + remote workers) or
vectorized stepping.

## 2. PPO trainer end-to-end

One real subprocess run through `LocalTrainingOrchestrator`
(512 env steps, rollout 256, epochs 2, batch 256, 1 env):

| metric | value |
|---|---|
| status | COMPLETED |
| total_timesteps | 512 |
| wall_clock_time | 2.2 s |
| SPS | 232.3 |

## 3. Batch scheduler dispatch

8 dummy jobs, max_workers=2:

| metric | value |
|---|---|
| create_batch | 1.91 ms |
| first tick (subprocess spawn) | 1.60 s |
| 8 jobs wall time | 9.7 s |
| status | completed |

Read: batch JSON persistence is ~2 ms; the dominant cost is subprocess
spawn (~0.8 s per launch). Scheduler bookkeeping overhead is negligible.

## 4. Curriculum controller

| metric | value |
|---|---|
| evaluate_advancement | ~2.6 µs/decision (1000 calls, 2.6 ms total) |

Curriculum advancement cost is negligible relative to an environment step
(~2.5 ms).

## Caveats

- Numbers are single-machine CPU measurements, not projections to GPU or
  cluster. No bitwise cross-machine reproducibility is claimed.
- PPO SPS includes subprocess startup amortized over a short 512-step run;
  longer runs amortize the ~1 s startup better.
- SAC/DQN SPS were not separately benchmarked; both complete smoke-scale
  runs correctly in the test suite (see PHASE_5_FINAL_REPORT §25).
