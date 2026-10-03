# AI Simulation Studio - Autonomous Vehicle RL Platform

A standalone 3D AI Environment Simulation Platform designed from the ground up for Reinforcement Learning (RL), autonomous vehicle training, and visual environment authoring.

The platform treats the vehicle, road geometry, sensors, observations, actions, rewards, and scenarios as **modular, data-driven components** rather than hard-coded racing game logic. It can be distributed as a standalone Windows executable requiring **zero runtimes, no Docker, and no development tools**.

---

## 1. Architectural Philosophy

The fundamental abstraction across the platform is:
```
Environment = World + Simulation + Agents + Sensors + Actions + Observations + Rewards + Termination + Scenarios
```

The simulator is strictly the **Environment**. The external AI is the **Agent**. The simulator remains completely decoupled from specific RL algorithms (PPO, SAC, DQN, TD3, etc.), communicating via high-performance network interfaces or standard Gymnasium environment contracts.

```
       RESET (seed, options)
                ↓
           OBSERVATION (kinematics, LiDAR, camera)
                ↓
         EXTERNAL AI MODEL (PPO / SAC / Neural Net)
                ↓
             ACTION (steering, throttle, brake)
                ↓
    PHYSICS STEP (fixed timestep, tire slip, drag)
                ↓
      NEW OBSERVATION + DECOMPOSED REWARD
                ↓
      TERMINATION / TRUNCATION / CONTINUE
```

---

## 2. Key Features

- **Visual Track & Road Authoring**:
  - Catmull-Rom parametric spline with uniform arc-length parameterization.
  - Variable road width per control point (not globally fixed).
  - 3D elevation and banking per segment.
  - Configurable road boundaries: guardrails, concrete walls, curbs, or open boundaries.
  - Real-time 3D procedural mesh generation (asphalt surface, UV mapping, dashed lane markings, curbs, barriers, checkpoints).
  - Clean separation between semantic logical road definitions and generated 3D geometry.

- **Modular Vehicle Dynamics**:
  - 4-wheel dynamic vehicle model with non-linear brush tire slip (Pacejka-style saturation).
  - Smooth low-speed kinematic to high-speed dynamic slip blending (eliminates $1/v_x$ singularities).
  - Aerodynamic drag, rolling resistance, steering rate limiting, and braking force.
  - Oriented Bounding Box (OBB) collision detection against track barriers and obstacles.

- **Modular Sensor Suite**:
  - Independent update frequencies (e.g., Physics at 60 Hz, Camera at 30 Hz, IMU at 100 Hz).
  - **Synthetic RGB Camera**: Offscreen Framebuffer (FBO) rendering from hood/roof perspective with procedural rasterizer fallback.
  - **Multi-Beam LiDAR / Rangefinder**: Configurable beam count (e.g., 15 rays), angular field of view, max range, and Gaussian noise.
  - **Vehicle State & Kinematics Sensor**: Speed, velocity components, accelerations, heading error, distance from centerline, and distance to next checkpoint.
  - **6-Axis IMU**: 3-axis accelerometer and 3-axis gyroscope with bias drift and noise.

- **Composable Reward Engine**:
  - Composable weighted terms: forward progress ($\Delta s$), centerline tracking ($1 - |d| / (w/2)$), speed target achievement, heading alignment ($\cos(\Delta \psi)$), action smoothness, checkpoint bonuses, and collision penalties.
  - **Full Reward Decomposition**: Every single reward component is explicitly broken down and returned in `info["reward_breakdown"]` and rendered in the live RL Debug Inspector.

- **Configurable Termination**:
  - Explicit reason attribution (`collision`, `off_road`, `wrong_direction`, `checkpoint_timeout`, `max_steps_exceeded`, `lap_completed`).

- **External AI Connection & Gymnasium Adapter**:
  - High-throughput TCP server with `TCP_NODELAY` framing NDJSON simulation packets.
  - Standard `SimGymEnv(gymnasium.Env)` wrapper allowing direct use with PyTorch, Stable-Baselines3, CleanRL, or custom loops.
  - Ready-to-run autonomous agents included:
    - `sim_client/agents/pid_driver.py`: Closed-loop lateral/longitudinal autonomous driver.
    - `sim_client/agents/random_agent.py`: Baseline exploration agent.
    - `sim_client/agents/ppo_train.py`: Self-contained PyTorch PPO training loop.

- **Episode Recording & Replay**:
  - Capture complete step trajectories (states, actions, decomposed rewards, collisions).
  - Replay player with play/pause, seek, and scrub controls.
  - Export/import to `.json` or `.json.gz`.

- **Domain Randomization & Scenarios**:
  - Seedable randomization of vehicle mass, tire friction, surface friction, sensor noise, and spawn poses.
  - Scenarios separating weather (clear, rain, fog), lighting (day, dusk, night), and obstacle configurations from physical road geometry.

- **Standalone Windows Distribution**:
  - Build script packages the application into a standalone folder with executable (`AI_Environment_Simulator.exe`) containing all necessary binaries and DLLs.

---

## 3. Directory Structure

```
Simulation/
├── sim_core/                      # Core simulation engine (decoupled from RL and UI)
│   ├── clock.py                   # Fixed timestep clock, determinism, seed
│   ├── math_utils.py              # Vec2, Vec3, OBB2D, SAT collision, raycasts
│   ├── track/                     # Road and spline geometry
│   │   ├── spline.py              # Catmull-Rom spline with arc-length parameterization
│   │   ├── road_definition.py     # Logical road control points, widths, elevations
│   │   ├── mesh_generator.py      # Procedural 3D mesh generator (road, curbs, barriers)
│   │   └── track_queries.py       # Spatial queries: centerline distance, lateral offset
│   ├── vehicle/                   # Modular vehicle physics
│   │   ├── vehicle_config.py      # Mass, geometry, tire friction, motor parameters
│   │   ├── vehicle_model.py       # 4-wheel dynamics, slip angles, drive/brake forces
│   │   └── collision.py           # OBB collision against barriers and obstacles
│   ├── sensors/                   # Modular sensor framework
│   │   ├── base_sensor.py         # Abstract sensor base with independent rates & noise
│   │   ├── vehicle_state_sensor.py# Telemetry & kinematics sensor
│   │   ├── raycast_sensor.py      # Multi-beam LiDAR rangefinder
│   │   ├── camera_sensor.py       # Synthetic RGB camera with offscreen buffer
│   │   ├── imu_sensor.py          # 6-axis IMU with bias drift and noise
│   │   └── sensor_manager.py      # Sensor update orchestration
│   └── world/                     # World entities & semantics
│       ├── entity.py              # WorldEntity with semantic labels
│       ├── obstacle.py            # Static obstacles (cones, barricades, barrels)
│       └── checkpoint.py          # Checkpoint gates and lap timing tracker
├── sim_env/                       # RL Environment layer (contract, rewards, spaces)
│   ├── environment.py             # SimulationEnvironment (reset/step lifecycle)
│   ├── spaces.py                  # ActionSpace and ObservationSchema
│   ├── reward_engine.py           # Composable reward system with decomposition
│   ├── termination_engine.py      # Termination & truncation evaluation
│   ├── domain_randomizer.py       # Seeded domain randomization
│   └── scenarios.py               # Scenario presets (weather, lighting, obstacles)
├── sim_render/                    # Modern 3D Graphics (ModernGL + PyOpenGL)
│   ├── renderer.py                # 3D Scene renderer (track, car, debug gizmos)
│   ├── camera.py                  # Chase, Hood, Top-down, Orbit 3D cameras
│   ├── shaders.py                 # GLSL shaders for road, vehicle, curbs, lines
│   ├── mesh.py                    # VBO / VAO mesh wrappers and 3D primitives
│   └── offscreen.py               # Offscreen FBO for synthetic camera sensor
├── sim_ui/                        # Visual Studio Editor & RL Debugger UI
│   ├── app.py                     # Main application entry point & event loop
│   ├── editor.py                  # Spline track authoring & control point manipulation
│   ├── ui_overlay.py              # 2D HUD texture blitting over OpenGL context
│   └── hud.py                     # Telemetry HUD, reward breakdown, camera PiP
├── sim_net/                       # High-performance external AI communication server
│   ├── protocol.py                # NDJSON protocol framing and message types
│   └── server.py                  # High-frequency TCP server with TCP_NODELAY
├── sim_client/                    # External Python SDK & Gym wrapper for external AI
│   ├── client.py                  # Lightweight TCP simulation client
│   ├── gym_env.py                 # Standard gymnasium.Env adapter
│   └── agents/                    # Ready-to-run external AI models
│       ├── pid_driver.py          # Autonomous closed-loop PID lane follower
│       ├── random_agent.py        # Baseline random exploration agent
│       └── ppo_train.py           # PyTorch PPO training script
├── sim_recorder/                  # Episode recording & replay system
│   ├── recorder.py                # Episode step recording & buffering
│   └── replay.py                  # Deterministic replay player
├── sim_project/                   # Project serialization & presets
│   ├── serializer.py              # Save / load *.sim.json environments
│   └── presets/                   # Built-in environments (oval, serpentine, obstacle)
├── build/                         # Standalone distribution packaging
│   └── package_windows.py         # PyInstaller automated packaging script
├── tests/                         # Comprehensive automated test suite (19 passing tests)
├── main.py                        # Root launcher executable entry point
└── presets/                       # Pre-generated environment presets
```

---

## 4. Running the Simulator

### Interactive 3D Studio Mode
Launch the interactive visual environment:
```bash
python main.py
```
Options:
- `--track oval` (default): Fast proving ground oval circuit.
- `--track serpentine`: Technical alpine track with elevation and variable road width.
- `--track obstacle`: Proving ground with obstacle barricades and cones.
- `--port 8765`: TCP port for external AI connections.

### Keyboard Controls
- **Drive**: `W` / `Up` (Throttle), `S` / `Down` (Brake), `A` / `D` or `Left` / `Right` (Steering), `Space` (Handbrake).
- **Reset Environment**: `R`
- **Cycle Cameras**: `C` (Chase $\rightarrow$ Hood $\rightarrow$ Top-Down $\rightarrow$ Orbit)
- **Toggle Observation Inspector**: `Tab`

### Headless Training Mode
For high-performance automated training or remote servers without a display:
```bash
python main.py --headless --port 8765
```

---

## 5. Connecting an External AI Model

### Option A: Using the Gymnasium Interface (`gymnasium.Env`)
Any standard RL library (PyTorch, Stable-Baselines3, CleanRL) can connect directly:
```python
from sim_client import SimGymEnv

env = SimGymEnv(host="127.0.0.1", port=8765)
obs, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()  # Or model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Inspect live decomposed rewards
    print("Reward breakdown:", info['reward_breakdown'])
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Option B: Running the Included Autonomous PID Driver
In a separate terminal while the simulator is running:
```bash
python -m sim_client.agents.pid_driver 127.0.0.1 8765
```
The PID agent connects over TCP, receives vehicle state and LiDAR range readings, modulates steering and throttle, and navigates the track autonomously!

### Option C: Running the PyTorch PPO Training Script
```bash
python -m sim_client.agents.ppo_train
```

---

## 6. Building the Standalone Windows Executable

To generate the standalone, zero-dependency Windows distribution package:
```bash
python build/package_windows.py
```
This produces `dist/AI_Environment_Simulator/` containing:
- `AI_Environment_Simulator.exe`
- All required OpenGL, SDL2, and Python runtimes pre-packaged.
- `presets/` directory with test tracks.

The user simply double-clicks `AI_Environment_Simulator.exe` to launch the simulator. **No Python, Docker, or manual installation required.**

---

## 7. Automated Test Suite

Run the complete test suite:
```bash
python -m pytest tests/ -v
```
Verified test coverage includes:
- Spline arc-length parameterization and projection queries.
- Procedural 3D track mesh generation and collision boundaries.
- Modular vehicle dynamics, tire slip, and braking physics.
- Sensor suite (LiDAR rays, camera FBO, IMU drift, state queries).
- Composable reward engine and component decomposition.
- Domain randomization seed reproducibility.
- Project save/load persistence (`*.sim.json`).
- High-frequency TCP protocol server/client handshake and stepping.
- Gymnasium integration compliance.
- Closed-loop autonomous driving loop verification.
