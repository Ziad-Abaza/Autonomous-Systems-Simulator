---
noteId: "4bffb330bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Phase 4 Performance

Measured with `benchmarks/benchmark_phase4.py` on the development machine
(Windows, Python 3.13, in-process envs, lane_following template, default
sensor suite). Reproduce with:

```powershell
python benchmarks/benchmark_phase4.py
```

Raw results: `benchmarks/phase4_results.json`.

## Environment step throughput

| Configuration | env-steps/s | ms/step |
|---|---|---|
| 1 env | 352.0 | 2.841 |
| 2 envs | 340.3 (170.2/env) | 2.938 |
| 4 envs | 332.4 (83.1/env) | 3.008 |

Aggregate throughput drops ~6% from 1→4 envs — CPU-bound; per-env speed
falls linearly as cores saturate. For wall-clock scaling use multiple
trainer processes or TCP mode across cores/machines.

## Sensor configuration impact

| Sensors | steps/s | ms/step |
|---|---|---|
| full suite (state + lidar + camera fallback + IMU) | 341.9 | 2.925 |
| lidar + state only | 364.6 | 2.743 |
| state only | 1393.8 | 0.717 |

The procedural headless camera fallback costs ~0.18 ms/step; the LiDAR
raycast is the dominant sensor cost (~2.2 ms/step). `state_only` shows
the physics+agent-pipeline floor (~0.72 ms/step).

## Metrics pipeline overhead

- Buffered `MetricsWriter` per-step `step`-scope writes: **−0.17%**
  (within noise; effectively zero) — 336.9 vs 336.3 sps.
- JSONL append + sanitize is amortized across the 64-record buffer.

## Orchestration overhead

- Contract build + validate: **0.05 ms** — negligible.
- Subprocess spawn (`python -m`): OS-level Python interpreter startup
  (~0.5–1 s) — one-time per run.
- `HeadlessSimProcessPool.start()`: bounded by pygame/sim import
  (~2–4 s per process) — only in TCP mode.

## PPO smoke run (148-test suite measurement)

- Orchestrated 256-timestep PPO run through subprocess completes in
  ~10–15 s including Python/torch startup and checkpoint writes.

## Conclusions

- The metrics/contract/orchestration layers add no measurable hot-path
  cost — step time is dominated by physics + sensors.
- Parallel in-process envs trade per-env speed for episode diversity
  rather than wall-clock gains; TCP mode is the isolation path.
