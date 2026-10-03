# Performance

Measured performance of Simulation Studio, with guidance for making training and interactive sessions faster.

> **Benchmark provenance:** numbers below are from the committed results in `benchmarks/*.json`, captured on the development machine — **Windows (win32), Python 3.13.7**. Treat them as indicative of that machine, not as guarantees on your hardware. Reproduce them with the scripts listed in [How to reproduce](#how-to-reproduce).

---

## Headline numbers

### Environment throughput (in-process stepping)

`benchmarks/benchmark_phase4.py` — steps `SimulationEnvironment` instances in-process, sequential stepping, full default sensor suite:

| Envs | Total env steps/s | Per-env steps/s | Avg step (ms) |
|---|---|---|---|
| 1 | 352.0 | 352.0 | 2.84 |
| 2 | 340.3 | 170.2 | 2.94 |
| 4 | 332.4 | 83.1 | 3.01 |

Interpretation: ~2.8–3.0 ms per `step()` with the full default sensor suite. Stepping multiple envs sequentially in one process is near-perfectly additive per env — total throughput stays ~330–350 steps/s — so single-process multi-env does not buy parallel throughput (use `process`/`tcp` env modes for that; see [Training env modes](../reinforcement-learning/training.md#environment-modes-env_mode)).

### Sensor cost

Same benchmark, different sensor suites (1500 steps each):

| Sensor suite | Steps/s | Avg step (ms) | vs. full suite |
|---|---|---|---|
| Full suite (vehicle_state + lidar 15-ray + camera 84×84 + IMU) | 341.9 | 2.93 | baseline |
| `lidar_only` | 364.6 | 2.74 | ~1.07× faster |
| `state_only` | 1393.8 | 0.72 | ~4.1× faster |

Takeaways:

- The **camera is the most expensive sensor** in the default suite — removing it (and IMU) recovers ~3 ms/step of overhead.
- **Lidar is cheap-ish but scales linearly**: it does an O(num_rays × boundary segments) linear scan — no broadphase. More rays (`num_rays` up to 64 in the inspector) or longer/more-segmented tracks cost proportionally more.
- State-only stepping is ~4× faster — if your agent doesn't need lidar/camera, strip the sensor suite (see `state8`-style observation presets in agentRL, or SENSORS tab toggles).

### Vision pipeline

`benchmarks/benchmark_vision.py` — 84×84×3 RGB frames:

| Metric | Value |
|---|---|
| Full 3D render + readback latency | 0.56 ms/frame → ~1786 fps theoretical |
| Render only (OpenGL) | 0.401 ms |
| Readback (preallocated `read_into`) | 0.159 ms |
| Env step with vision obs | 0.576 ms → **1736 steps/s** |
| Env step, vector-only | 0.561 ms → 1783 steps/s |

Note: the vision env-step numbers were measured on the **procedural fallback** camera path (`CameraSensor._synthesize_perspective_road`), which is much cheaper than a real 3D render — in the interactive studio, camera frames additionally cost a real offscreen FBO render + readback per sensor tick. Even so, at 30 Hz sensor rate against a 60 Hz env, per-step vision overhead is small in this configuration.

### Collision broadphase (spatial hash)

`benchmarks/benchmark_broadphase.py` — `SpatialHashGrid2D` (cell size 12 m) over track boundary segments:

| Track | Length | Segments | Speedup | Candidate reduction | Correctness |
|---|---|---|---|---|---|
| Small | 264 m | 528 | 2.35× | 96.0% | 100% |
| Medium | 819 m | 1638 | 5.75× | 98.8% | 100% |
| Large | 2363 m | 4726 | 15.31× | 99.6% | 100% |

Vehicle↔boundary collision is effectively **constant-time per step** regardless of track size (~40 µs with the grid vs. up to ~626 µs brute-force). Zero false negatives; memory overhead is tens of KB.

Scope note: the broadphase is used for **vehicle collision only**. Lidar raycasts and `get_closest_point` remain O(N) linear scans, so very long tracks still raise lidar and track-query cost per step.

### Metrics overhead

Same benchmark file: 336.3 steps/s without metrics vs. 336.9 with metrics — **~0% overhead** (within noise) for `metrics.jsonl` recording, thanks to buffered writes (64 records/flush).

---

## Headless multi-env server ceiling

`SimServerMulti` (`main.py --headless --num-envs N`) runs a single-threaded `select` loop that sleeps **0.0005 s per poll** → a shared ceiling of roughly **2000 socket polls/s across all connected clients**. Environments only advance on `STEP` requests (lockstep), so total step throughput is bounded by both that poll rate and the per-step compute (~3 ms with the full suite).

With `--num-envs 1`, headless mode goes through `SimulationStudioApp(headless=True)` + single `SimulationServer`, which sleeps 0.005 s per loop instead — lower ceiling, still fine for a single agent.

---

## Practical guidance

### For training

- **Run headless.** Rendering is the largest avoidable cost. Use `python main.py --headless`, or `env_mode=process`/`tcp_multi` in experiments (orchestrator spawns headless sims for you).
- **Slim the sensor suite.** Drop the camera if your policy doesn't use images (~4× step speedup when going state-only). Reduce lidar `num_rays`/`max_range` for marginal gains.
- **Slim the observation.** Smaller vectors step marginally faster and make every consumer (network, buffers) cheaper — e.g. agentRL's `state8` preset (8 dims) vs `full23`.
- **Choose an env mode:**

  | `env_mode` | How it works | Trade-off |
  |---|---|---|
  | `inprocess` | Envs in the trainer process | No IPC; sequential stepping — total sps flat ~330–350 regardless of `num_envs` |
  | `process` | One spawned `mp.Process` per env (`ProcessVectorEnv`, pipelined pipes) | Real CPU parallelism; IPC serialization per step; worker crash → `EnvWorkerCrash` (retryable) |
  | `tcp` | N separate `main.py --headless` processes, one port each | Parallelism + process isolation; TCP + NDJSON overhead; 30 s per-port startup timeout |
  | `tcp_multi` | One `main.py --headless --num-envs N` process, one port | Single extra process; shared ~2000 polls/s ceiling; FIFO env-slot assignment |

  `process`/`tcp` modes pay per-step serialization to gain parallel stepping — worth it when per-step compute or `num_envs` is high. `inprocess` wins for small `num_envs` and cheap envs.

### For interactive studio

- Shrink the window: `--width 960 --height 540` (default 1280×720).
- Disable or shrink the camera sensor (resolution 32–512 px, rate in the SENSORS tab); camera PiPs each cost offscreen renders.
- Lower `ui_scale` (SETTINGS; 0.9 is the minimum offered by the UI buttons; the setting clamps to [0.75, 2.0]).
- Termination timeouts are disabled in interactive mode (`max_episode_steps=0`, `max_seconds_without_checkpoint=0`) — that's intentional, not a perf bug.

---

## How to reproduce

| What | Command |
|---|---|
| Env throughput + sensor cost + metrics overhead | `python benchmarks/benchmark_phase4.py` |
| Broadphase spatial hash | `python benchmarks/benchmark_broadphase.py` |
| Vision pipeline (needs a GL context) | `python benchmarks/benchmark_vision.py` |
| Multi-env throughput via the experiment CLI | `python -m sim_experiment.cli benchmark --envs 1,2,4 --steps 2000` |
| Env-mode scaling (inprocess/process/tcp) | `python benchmarks/phase6/perf_runner.py` |
| Vehicle dynamics maneuver suite (physics-only, dt = 1/60) | DYNAMICS tab in the studio, or `tools/vehicle_dynamics` |

Results are written next to the scripts (e.g. `benchmarks/phase4_results.json`, `broadphase_results.json`, `vision_results.json`).

---

## See also

- [Troubleshooting → Performance problems](../troubleshooting/troubleshooting.md#performance-problems)
- [Sensors](../sensors/sensors.md) — per-sensor parameters that drive cost
- [Architecture](../architecture/architecture.md) — where stepping, sensors, and rendering live
- [Training](../reinforcement-learning/training.md) — `env_mode` selection for runs
