---
noteId: "ea90ddb0bed511f1a29f1fbaabbd87c8"
tags: []

---

# Phase 2 Engineering Plan: AI Environment Authoring Studio & Scalable Simulation

**Author:** Antigravity Engine Architecture Team  
**Date:** October 2026  
**Target:** AI Environment Simulation Studio — Phase 2 Implementation  

---

## 1. Current Architecture Relevant to Phase 2

The simulation foundation hardened in Phase 1 provides:
* **Physics & Vehicle Dynamics (`sim_core/vehicle/`)**: 60 Hz deterministic integration, 4-wheel slip angle dynamics with Pacejka tire modeling, SAT-based oriented bounding box collision.
* **Track Representation & Spline (`sim_core/track/`)**: Catmull-Rom spline with arc-length parameterization, heading error and centerline queries, procedural mesh generation with curbs, barriers, and checkpoint gates.
* **Collision Detection (`sim_core/vehicle/collision.py`)**: Tests vehicle OBB against all boundary segments ($O(N)$ linear loop over 600–2000 segments) and obstacles.
* **Sensors (`sim_core/sensors/`)**: Modular sensor suite (VehicleState, Raycast LiDAR, IMU, Camera) orchestrated by `SensorManager`.
* **RL Engine & Environment (`sim_env/`)**: Gymnasium-compatible `SimulationEnvironment`, composable `RewardEngine`, attributed `TerminationEngine`, `DomainRandomizer`, `ObservationSchema`, `ActionSpaceConfig`.
* **Authoring & UI (`sim_ui/`)**: Pygame + ModernGL application with 2D spline editor (`editor.py`), `EnvironmentInspector`, top/bottom status HUDs.
* **Persistence (`sim_project/`)**: `EnvironmentProject` (Schema 1.0.0) saving/loading tracks, vehicles, and configurations.
* **Networking & Client (`sim_net/`, `sim_client/`)**: Non-blocking NDJSON TCP server and client, Gymnasium adapter `SimGymEnv`.

---

## 2. Existing Capabilities

1. **Deterministic 60 Hz Simulation**: Verified physics stepping and sensor sampling.
2. **Spline Track Generation**: Generates 3D surface ribbons, curbs, wall barriers, and collision segments from 2D/3D control points.
3. **Decomposed Reward Computation**: Real-time calculation of forward progress, centering, target speed, heading alignment, steering smoothness, checkpoint bonuses, and collision penalties.
4. **Basic Spline Editing**: Mouse-based point addition, dragging, point deletion, and spawn placement.
5. **Inspectable Parameters**: Inspector tabs for track boundaries, selected point width/elevation, spawn position, and vehicle mass/power.
6. **Project Persistence**: Serialization to `.sim.json` for tracks, vehicles, and environment configs.
7. **Regression Suite**: 32/32 tests passing cleanly in `tests/`.

---

## 3. Missing Capabilities

1. **Interactive Authoring & Scene Composition**:
   - No interactive insertion of control points between existing segments.
   - Limited visual gizmos: lack of curvature visualization, bank angle indicators, tangent vectors, width preview handles, and checkpoint/spawn orientation gizmos.
   - No interactive placement, rotation, scaling, or editing of environment entities (obstacles, barriers, cones, signs, traffic lights).
   - No unified scene hierarchy / entity list with bidirectional selection (Canvas $\leftrightarrow$ Object List).
2. **Reward Configuration UI & Runtime Breakdown**:
   - Reward weights and penalties are hardcoded or CLI-configured; not dynamically inspectable or editable in authoring.
   - Lack of parameter validation on reward modifications to prevent invalid contracts.
3. **Sensor Authoring & Observation Inspection**:
   - Inability to add, remove, enable/disable, or customize individual sensors in the editor.
   - Lack of explicit visual distinction between agent observations (AI perception) and debug/oracle telemetry (ground truth).
4. **Spatial Acceleration Structure (Broadphase Collision)**:
   - Collision detection currently performs an $O(N)$ linear scan over all boundary segments.
   - Longer tracks or multi-obstacle scenes experience proportional collision query slowdowns.
5. **Vision Pipeline & Camera Isolation**:
   - `glReadPixels` synchronously halts the GPU pipeline during offscreen FBO readbacks.
   - Need measured comparison between procedural rasterizer and GPU transfer paths, with strict logical isolation between Human View, Agent Camera, and Debug Camera.
6. **RL Baseline Training & Benchmarking**:
   - `sim_client/agents/ppo_train.py` was a simple rollout loop without PPO clipping, GAE, or loss backpropagation.
   - Need a complete, reproducible PPO training benchmark execution recording episode return, lateral error, collision rate, loss metrics, and steps/sec throughput.
7. **Project Schema Evolution (v2.0.0)**:
   - Schema versioning must support new entity lists, obstacle properties, custom sensor suites, and reward configs with backward compatibility for v1.0.0.

---

## 4. Dependencies Between Tasks

```
[2A.3 Entity System & 2A.4 Inspector] ──┐
                                       ├──> [2A Scene Hierarchy & Editor Gizmos] ──> [2H Save/Load v2.0]
[2B Reward UI & 2C Sensor Config] ────┘                                                     │
                                                                                           ▼
[2D Spatial Hash Broadphase] ──────────> [Broadphase Benchmark] ───────────────────> [2E Vision Benchmark]
                                                                                           │
                                                                                           ▼
                                                                                   [2F RL PPO Baseline]
                                                                                           │
                                                                                           ▼
                                                                                   [Final Validation]
```

---

## 5. Implementation Order

1. **Step 1: Entity & World Model Extension (`sim_core/world/`)**
   - Generalize `WorldEntity` with reusable entity definitions: `StaticObstacle`, `Barrier`, `TrafficCone`, `TrafficSign`, `TrafficLight`, `CheckpointGate`, `SpawnPoint`.
   - Update `RoadDefinition` and `EnvironmentProject` to host scene entities.
2. **Step 2: Spatial Acceleration Broadphase (`sim_core/collision/` & `sim_core/vehicle/`)**
   - Implement `SpatialHashGrid2D` broadphase acceleration structure.
   - Integrate with `VehicleCollisionChecker` and raycasts.
   - Preserve zero false negatives and exact narrowphase intersection correctness.
   - Create independent broadphase test and benchmark script (`benchmarks/benchmark_broadphase.py`).
3. **Step 3: Interactive Editor & Visual Gizmos (`sim_ui/editor.py`, `sim_ui/inspector.py`)**
   - Add control point insertion on spline segments, road width handles, banking preview, elevation slope cues, and curvature visualization.
   - Implement interactive entity placement tool, entity gizmos (rotation handle, selection box).
   - Add Scene Hierarchy / Entity Tree list in UI with bidirectional selection.
4. **Step 4: Reward & Sensor Configuration Workflows (`sim_ui/inspector.py`, `sim_env/`)**
   - Add Reward Configuration tab to Inspector with parameter clamping and validation.
   - Add Sensor Suite Manager tab to Inspector: add/remove sensors, configure noise/latency/resolution, preview observation schema.
   - Build Observation Preview panel clearly separating agent perception from oracle telemetry.
5. **Step 5: Vision Pipeline Optimization & Isolation (`sim_render/`, `sim_core/sensors/`)**
   - Optimize frame readback pipeline and profile GPU vs procedural rasterization.
   - Ensure complete logical separation between Human View, Agent Camera, and Debug Camera.
   - Create vision pipeline latency benchmark (`benchmarks/benchmark_vision.py`).
6. **Step 6: Project Persistence & Schema Migration (`sim_project/`)**
   - Upgrade `EnvironmentProject` to schema `2.0.0`.
   - Implement migration handler for `1.0.0` files.
   - Verify complete save $\rightarrow$ load $\rightarrow$ rebuild $\rightarrow$ run cycle.
7. **Step 7: RL Baseline Training Benchmark (`sim_client/agents/ppo_baseline.py`)**
   - Implement complete PyTorch PPO algorithm (clipped surrogate objective, GAE advantage estimation, value function loss, entropy bonus).
   - Configure reproducible training environment baseline (`experiments/baseline_ppo/`).
   - Run training run, record throughput, episode metrics, loss curves, and save model checkpoint.
8. **Step 8: Automated Regression Testing & Manual Runtime Validation**
   - Maintain all 32 existing tests.
   - Add comprehensive tests for authoring, broadphase, sensors, reward, schema migration, and PPO.
   - Execute manual verification of editor workflows and simulation stability.
9. **Step 9: Complete Documentation & Final Report**
   - Generate all required documents: `AUTHORING.md`, `ENTITY_MODEL.md`, `SENSOR_CONFIGURATION.md`, `REWARD_CONFIGURATION.md`, `PERFORMANCE_BENCHMARKS.md`, `RL_BASELINE.md`.
   - Compile comprehensive 14-section Final Report.

---

## 6. Validation Strategy

* **Unit & Subsystem Testing**: Run `python -m pytest` after every subsystem modification.
* **Broadphase Correctness**: Directly verify that broadphase queries yield identical collision results as exhaustive $O(N)$ narrowphase scans across 10,000 randomized test poses on multiple track configurations.
* **Persistence Integrity**: Test that saving, reloading, and rebuilding any environment preserves all geometry, control points, entities, sensor properties, and reward parameters bit-for-bit.
* **Observation Isolation**: Verify that agent observation vectors contain zero privileged state when telemetry flags are disabled.
* **RL Stability**: Verify that the PPO training loop steps continuously without NaN propagation, memory leaks, or socket desynchronization.

---

## 7. Performance Measurement Strategy

* **Broadphase**: Measure microsecond latency and candidate segment counts on Small (oval, 150m), Medium (circuit, 600m), and Large (endurance, 2500m) tracks. Compare $O(N)$ brute-force vs `SpatialHashGrid2D`.
* **Vision Pipeline**: Profile render call time, buffer transfer time, and end-to-end environment step time for procedural rasterizer vs offscreen FBO.
* **Simulation Throughput**: Measure environment steps per second (SPS) during headless RL stepping and multi-episode training.

---

## 8. Risks & Mitigations

1. **Risk: Broadphase Misses Borderline Collisions (False Negatives)**
   * *Mitigation*: Ensure cell query bounding box expands by maximum vehicle radius ($r = \sqrt{w^2 + l^2}/2$) plus safety envelope, covering all potential candidate segments.
2. **Risk: OpenGL Context Conflicts During Headless Training**
   * *Mitigation*: Fall back gracefully to procedural perspective rasterization whenever headless mode or no OpenGL context is active, maintaining consistent observation dimensions and channel layouts.
3. **Risk: Schema Incompatibility with Phase 1 Presets**
   * *Mitigation*: Version migration logic detects schema `"1.0.0"`, parses legacy attributes, and assigns sensible defaults for new v2.0 fields without raising parse errors.
4. **Risk: PPO Training Wall-Clock Time in Single Process**
   * *Mitigation*: Run a verified deterministic training benchmark with tuned batch size and vectorized PyTorch tensor operations to ensure fast convergence and high throughput.

---

## 9. Definition of Done

* **Authoring**: Full 2D track authoring workflow with interactive point insertion, road width/elevation/banking gizmos, entity placement (obstacles, barriers, cones, signs, traffic lights), unified scene hierarchy, and bidirectional selection.
* **Reward Engine**: Expose, validate, and dynamically configure all reward components with real-time UI breakdown.
* **Sensor Suite**: Modular sensor authoring, inspectable observation schema, and clean separation between agent observations and oracle diagnostics.
* **Simulation Scalability**: 2D Spatial Hash broadphase integrated, tested, and benchmarked with proven speedup on large tracks and zero false negatives.
* **Vision System**: Measured camera pipeline with isolated Human, Agent, and Debug camera feeds.
* **RL Benchmark**: Reproducible PPO baseline experiment executed with recorded returns, loss metrics, and checkpoints stored in `experiments/baseline_ppo/`.
* **Persistence**: Schema 2.0.0 persistence with backward compatibility for 1.0.0.
* **Quality**: All existing 32 tests stay green + new unit/integration test suites passing. Comprehensive documentation and Final Report delivered.
