# Simulation Studio Timing & Determinism Model

## Overview

The AI Simulation Studio decouples the physics clock, sensor update schedules, rendering frame rates, external AI inference cycles, and networking layers. This ensures training validity, numerical stability, and deterministic replayability across both interactive 3D and headless execution.

```
+-------------------------------------------------------------------------+
|                          External AI Client                             |
|         (Gymnasium / Stable-Baselines3 / CleanRL / PyTorch)             |
+-------------------------------------------------------------------------+
                                    |
                           TCP (NDJSON / TCP_NODELAY)
                                    |
+-------------------------------------------------------------------------+
|                           Simulation Server                             |
|          Lockstep Request-Response (RESET, STEP, GET_STATE)             |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                         Simulation Environment                          |
|                                                                         |
|   1. Action Validation & Decoding (ActionSpaceConfig)                   |
|   2. Fixed Physics Stepping (FixedClock: dt = 1 / 60.0s)                |
|      - Dynamic 4-wheel tire slip + low-speed kinematic blend            |
|      - Semi-implicit Euler numerical integration                        |
|   3. Spatial Geometry & Collision Tests (SAT OBB + Segments)            |
|   4. Directional Checkpoint Tracking                                    |
|   5. Decomposed Reward Computation (Clamped Progress, Centering, etc.)  |
|   6. Termination & Truncation Evaluation (Explicit Cause Attribution)   |
|   7. Sensor Scheduling & Latency Buffering                              |
|      - VehicleStateSensor @ 60 Hz                                       |
|      - RaycastSensor (LiDAR) @ 30 Hz                                    |
|      - CameraSensor (RGB) @ 30 Hz                                       |
|      - IMUSensor @ 60 Hz                                                |
|   8. Observation Assembly (ObservationSchema Isolation)                 |
+-------------------------------------------------------------------------+
            |                                         |
            v (Interactive Mode Only)                 v (Headless Mode)
+-------------------------+               +-------------------------------+
|   ModernGL 3D Renderer  |               |  Immediate Loopback Response  |
|   Pygame SDL2 (60 FPS)  |               |  (Up to 670 steps/sec)        |
+-------------------------+               +-------------------------------+
```

---

## 1. Timing Domains

| Domain | Frequency / Step | Mechanism | Coupling Status |
| :--- | :--- | :--- | :--- |
| **Physics Step** | Fixed 60.0 Hz ($\Delta t = 16.667\text{ ms}$) | [`FixedClock`](file:///D:/coding/projects/Simulation/sim_core/clock.py) semi-implicit Euler | **Strictly Decoupled** from render and wall clock |
| **Render Frame** | VSync / 60 FPS (interactive) or 0 (headless) | [`SimulationRenderer3D`](file:///D:/coding/projects/Simulation/sim_render/renderer.py) ModernGL loop | **Decoupled**; does not advance simulation clock |
| **Sensor Sampling** | Independent per sensor (1.0 Hz – 120.0 Hz) | [`BaseSensor.should_update`](file:///D:/coding/projects/Simulation/sim_core/sensors/base_sensor.py#L41) | **Decoupled**; samples cached between updates |
| **External AI** | Lockstep per step action | TCP `STEP` $\to$ `STEP_ACK` | **Lockstep**; advances physics by exactly 1 step |
| **Sensor Latency** | $0.0\text{ s} \le \tau \le 0.5\text{ s}$ | History queue indexing: $t_{target} = t_{sim} - \tau$ | Simulates real-world sensor pipeline delays |

---

## 2. Decoupling Guarantee

1. **Headless Execution**:
   In headless mode (`--headless`), the graphics pipeline and rendering loops are bypassed. Each incoming `STEP` command from an external agent advances the simulation by exactly one fixed physics timestep $dt = 1/60\text{ s}$. The environment step rate is bounded only by CPU compute and network latency (~670 steps/sec).

2. **Sensor Decoupling**:
   Sensors maintain an internal `update_interval = 1.0 / update_frequency_hz`. If physics steps at 60 Hz and LiDAR is configured at 30 Hz:
   * Step $k=0$ ($t=0.000\text{s}$): LiDAR computes fresh 15-beam raycasts.
   * Step $k=1$ ($t=0.016\text{s}$): LiDAR interval has not elapsed; returns cached measurement from $t=0.000\text{s}$.
   * Step $k=2$ ($t=0.033\text{s}$): LiDAR updates and produces fresh measurement.
   Observation vector dimensions remain strictly invariant across all steps.

3. **Determinism**:
   Given identical random seed $S$ and identical action sequence $A_0, A_1, \dots, A_N$:
   * Simulation state trajectories $X_0, X_1, \dots, X_N$ are bitwise deterministic.
   * Domain randomization parameters are deterministically generated via `np.random.default_rng(seed)`.
