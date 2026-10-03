---
noteId: "e197a220bf6111f1a29f1fbaabbd87c8"
tags: []

---

# AGENT_RL Architecture Audit

**Date:** 2026-10-03
**Scope:** Full audit of the Simulation Studio repository as the foundation for a
continual multi-track RL driving agent in `agentRL/`.
**Method:** Static inspection of every relevant module + review of all recorded
training results in `experiments/`. File:line citations throughout.

> **Location note:** the task spec asks for `docs/AGENT_RL_*.md`, but the repo's
> `.gitignore:90` pattern `docs/` matches **any** directory named `docs` at any
> depth ("Internal docs & dev evidence") and is blocked for agent file tools.
> Agent documentation therefore lives at the root of `agentRL/` — tracked,
> self-contained with the deliverable, same filenames.

---

## 0. Executive Summary

The repository is a **mature simulation + experiment platform** (7 completed
phases) with almost everything an RL agent needs *except a working learned
policy*. The infrastructure is genuinely good: a versioned TCP protocol, a
declarative observation/action/reward/termination contract, domain
randomization, curriculum runtime, vector/process envs, checkpointed from-scratch
PPO/SAC/DQN/BC trainers, datasets, evaluation, orchestration, and remote workers.

**The single most important finding:** every recorded training run produced a
**degenerate stationary policy**. All episodes run to timeout at near-zero
speed, collecting dense "alive" rewards (centering + heading ≈ +0.8/step for
free). Zero collisions are trivially achieved by not moving. Nothing in this
repo has demonstrably learned to drive. This is a **reward-design defect**, not
a platform defect — and it is the first thing `agentRL` must fix via reward
configuration.

**Second key finding:** `agentRL/` exists but is **empty**. `tests/agent/`
exists and is empty. There is **no multi-track or continual-learning machinery
anywhere** — curricula vary scenario parameters on a single track only.

Runtime: Python 3.13.7, PyTorch 2.10.0 **CPU-only** (no CUDA), gymnasium 1.2.1,
no stable-baselines3. All existing RL code is from-scratch PyTorch.

---

## 1. Simulator Architecture Overview

```
sim_core/     vehicle dynamics, spline/track, sensors, collision, world entities
sim_env/      RL environment contract — spaces, obs/action/reward/termination
              designers, scenarios, randomization, curriculum defs, validator
sim_net/      TCP protocol (NDJSON) + single/multi-env servers
sim_client/   Python SDK, gymnasium adapter, reference agents (pid/random/PPO/SAC/DQN)
sim_experiment/ experiment manifests, runs, trainers, vec envs, datasets,
                evaluation, orchestration, remote workers
sim_render/   ModernGL renderer + offscreen sensor FBO
sim_ui/       Simulation Studio (track editor, inspector, replay, datasets)
sim_project/  *.sim.json documents, track library, settings
sim_recorder/ episode recording + replay
agentRL/      (EMPTY — the deliverable of this task)
tests/agent/  (EMPTY — designated experimentation laboratory)
```

Environment projects are `.sim.json` files (`schema_version` 2.0.0/3.0.0)
containing `road_definition`, `vehicle_config`, `agent` (declarative RL
contract), `scenario_def`, `episode_config`, `entities[]`, `curriculum`,
`fingerprint`, and legacy config blocks. `tracks/` holds 12 such files
(ovals, serpentine, smoke tests, several untitled editor outputs).

---

## 2. Environment Core (`sim_env/environment.py`)

`SimulationEnvironment` (line 41): Gymnasium-API-compatible (`reset()`,
`step()` 5-tuple) but **not** a `gym.Env` subclass; no `gym.spaces` objects —
spaces are schema dataclasses compiled from an `AgentDefinition`.

- **Timestep:** fixed 60 Hz env step (`FixedClock`, dt = 1/60 s); vehicle
  physics sub-stepped ×4 internally → effective 240 Hz integration
  (`sim_core/vehicle/vehicle_model.py:125-128`, `sim_core/clock.py`).
- **Dual pipeline:** legacy engines (`RewardEngine`, `TerminationEngine`,
  `DomainRandomizer`, legacy obs/action schemas) coexist with Phase-3 compiled
  pipelines (`CompiledActionDecoder`, `CompiledObservationPipeline`,
  `CompiledRewardEngine`, `CompiledTerminationEvaluator`). When `env.agent` is
  set, compiled pipelines win (env.py:90-105, 150-156). `agentRL` uses the
  declarative path exclusively.
- **step() order** (env.py:383-554): validate/decode action → vehicle step →
  entity updates → collision checks → track query (s, lateral_offset,
  heading_error, is_on_road, road_width) → checkpoint tracker → reward →
  termination/truncation → duration-limit → sensor updates → obs + info.
- **reset()** (233-365): seeds `clock.rng`; samples domain randomization
  (scenario-def version preferred); applies mass/tire/surface-friction and
  sensor-noise multipliers (idempotent via snapshot); clears
  `_scenario_spawned` entities; computes spawn pose (spawn_override >
  custom pose > road spawn + jitter); resets trackers/engines/sensors.
- **Step-after-done** returns `(last_obs, 0.0, True, True, reason=
  "invalid_call_after_done")` — ambiguous terminated+truncated both-true
  contract (env.py:372-381). `agentRL` treats either flag as done.
- **info dict** (env.py:645-680): `step, sim_time, terminated, truncated,
  termination_reason, termination_info, reward_breakdown, total_reward, speed,
  lateral_offset, heading_error, road_width, is_on_road, is_colliding,
  action_valid, action_error, checkpoints_passed, current_checkpoint,
  laps_completed, lap_progress, last_action`.

### Dormant/unimplemented config fields (evidence: env.py)
- `EpisodeConfiguration.max_steps` (default 3600) never enforced — only
  `max_duration_seconds` + termination MAX_STEPS rule are.
- `SpawnMode.RANDOM_CHECKPOINT` declared, not implemented (env.py:296-309).
- `ResetBehavior`, `auto_reset_on_done`, `AgentSpawnConfig` — config-only.

---

## 3. Observation System

### Contract & security (`sim_env/observation_contract.py`, `observation_designer.py`)
Every field has an authoritative classification:
`AGENT_OBSERVATION` (legal for policy) / `DEBUG_TELEMETRY` (forbidden) /
`ORACLE_GROUND_TRUTH` (forbidden) / `DIAGNOSTIC` (GET_STATE + info only).
`validate_no_leakage()` enforced at pipeline compile (raises `ValueError`,
observation_designer.py:334) and re-checked by `validator.py:231-234`.
This is the anti-cheat backbone `agentRL` builds on — the policy consumes
only AGENT_OBSERVATION channels.

### Default vector observation (23-dim float32, all vehicle_state-sourced)
| idx | channel | normalization |
|---|---|---|
| 0 | speed | /45 m/s |
| 1-2 | vel_body (x,y) | /45, /10 |
| 3 | yaw_rate | /3 rad/s |
| 4 | steering_angle | /0.6 rad |
| 5 | distance_from_center | /half road width |
| 6 | heading_error | /π |
| 7 | distance_to_checkpoint | /100 m capped |
| 8-22 | lidar_ranges ×15 | [0,1] |

**Oracle caveat:** channels 0-7 are fed by `VehicleStateSensor` — ground-truth
kinematics/track state with *configurable* noise. They are privileged
information deliberately whitelisted as AGENT_OBSERVATION. Only `lidar_ranges`
and camera images are genuinely sensor-derived. For honest claims about
"learned from observations," `agentRL` experiments label which observation
sets are state-oracle vs. sensor-derived (ablation axis).

### Camera
`image_channels` config → `{"vector":…, "image":…}` dict obs; 84×84×3 uint8.
Sensor: `CameraSensor` 84×84 RGB, 75° FOV, 30 Hz, mount (1.0,0,1.1),
additive Gaussian noise. **Fallback** when no GL renderer: a procedural
flat-shaded trapezoid rasterizer (`camera_sensor.py:73-144`) — poor fidelity
for pixel-based RL in headless training; real offscreen FBO only in rendered
mode.

### Temporal
**No frame stacking, no history, no recurrent support anywhere** — the only
temporal mechanism is server-side action rate limiting.

---

## 4. Action System

Declarative `ActionSpaceDefinition` (`sim_env/action_designer.py:72`):
3 continuous channels, normalized units:
- steering [-1,1], dead_zone 0.02, rate_limit 4.0 /s
- throttle [0,1], dead_zone 0.01, rate_limit 5.0 /s
- brake    [0,1], dead_zone 0.01, rate_limit 6.0 /s

`CompiledActionDecoder` applies scaling → dead-zone → per-channel rate limiting
against `prev_action` using real dt → clamp. Discrete variant: 5 presets
(Coast/Accelerate/Brake/Left/Right). NaN→0, Inf→clamped. Vehicle dynamics
further rate-limit steering at 4.5 rad/s (vehicle_model.py:146-148).

Implication: the **effective action contract is already smoothed** — policies
output normalized commands; the env enforces physical rate limits. `agentRL`
does not double-clip; it passes policy actions through and relies on
env-side validation (`action_valid`/`action_error` in info).

---

## 5. Reward Pipeline

Two implementations with identical math:
- Legacy `RewardEngine` (`sim_env/reward_engine.py`) — named weights.
- Compiled `CompiledRewardEngine` (`sim_env/reward_designer.py`) — component
  list `{component_type, weight}`, weighted-sum aggregation, closed type set.

Components: `progress` (Δs, +1.0), `centerline` (falloff, +0.5), `speed`
(min(1,v/20), +0.2), `heading` (cos θ, +0.3), `smooth_steer` (−Δ², −0.05),
`checkpoint` (+10/gate), `completion` (+100), `collision` (−50), `off_road`
(−25), `reverse` (−1), `time_penalty` (−0.01/step). Full decomposition per
step in `info["reward_breakdown"]`. Anti-hack guards already present:
5 m/step teleportation clamp on Δs, wrap-around fix, reverse-driving penalty,
directional checkpoint gates (`sim_core/world/checkpoint.py:51-58`).

### Reward-hacking analysis (confirmed by recorded results)
A stationary, centered, track-aligned car earns:
`centering(0.5) + heading(0.3) = +0.80/step` risk-free.
Driving at target 20 m/s earns `progress(0.33) + speed(0.2)` ≈ +0.53 extra —
only ~66% more, while exposing the agent to −50 collision termination.
Early in training, PPO's random exploration produces collisions; the safe
gradient is toward *not moving*. All recorded runs converged to exactly this
local optimum (§12). **`agentRL` must rebalance rewards** (gate centering
/heading terms on speed, raise progress weight, enable `time_penalty` and
`checkpoint_timeout` truncation) — all achievable via config, no engine
changes required.

---

## 6. Termination / Truncation

`sim_env/termination_designer.py` — declarative rules, first-triggered wins,
strict `(terminated, truncated, reason_dict)` with correct Gymnasium
semantics (`is_truncation` flag):
- `collision`, `off_road`, `wrong_direction` (>120°) → terminate
- `max_steps` (default 5000), `simulation_timeout` (60 s),
  `checkpoint_timeout` (30 s w/o gate) → truncate
- `course_completion` (target_laps) → terminate; `custom_threshold` declared
  but **not implemented**.

Env-level: `scenario_def.time_limit_override` >
`episode_config.max_duration_seconds` (default 60 s) → truncate
`max_duration_exceeded`. `checkpoint_timeout` is the built-in "stuck
detector" that counters the stationary-policy exploit when enabled.

---

## 7. Scenario, Obstacles & Track Variation

`sim_env/scenarios.py` + `scenario_designer.py`: `ScenarioDefinition` =
weather, time_of_day, ambient_light, `surface_friction_mult`,
`target_speed_override`, `time_limit_override`, `sensor_noise_mult`,
`spawn_override {pos, yaw_deg, initial_speed}`, `obstacle_overrides`,
embedded `randomization`. 6 presets: `basic_lane_following`,
`high_speed_racing`, `wet_adverse_weather`, `obstacle_evasion`,
`sensor_noise_challenge`, `full_domain_randomization`.

**Obstacles:** spawned at reset from `obstacle_overrides` entries
`{name, entity_type, pos, yaw}` → `create_entity`, tagged `_scenario_spawned`
(recreated each reset, never accumulate). `SET_SCENARIO` (protocol 2.1) can
swap scenario mid-connection. **Critical gap: obstacle placement is fully
deterministic per scenario — no randomized obstacle positions/counts.**
Learned-avoidance training needs per-episode randomized obstacle placement —
generated client-side via `SET_SCENARIO` (no simulator changes needed).

**Entities** (`sim_core/world/entity.py`): `StaticObstacle` (box/barrel/crate/
rock), `Barrier` (concrete/guardrail/tire_wall), `TrafficCone`, `TrafficSign`,
`TrafficLight`, `CheckpointEntity`, `SpawnEntity` — all collidables expose
OBB + boundary segments. Collision **detection only — no response**;
`is_colliding` flag only (`vehicle/collision.py`).

**Track variation:** `env.set_road_definition()` + rebuild; tracks loaded from
`.sim.json`. Multi-track training = multiple env instances or per-episode
project swap. Track geometry itself is never part of obs (only relative
quantities) — generalization must come from geometry-agnostic obs (lateral
offset, heading error, lidar).

---

## 8. Domain Randomization

`DomainRandomizationDefinition` (`sim_env/randomization_designer.py`):
per-parameter FIXED/UNIFORM/NORMAL + clipping, sampled per reset via seeded
`clock.rng`. **Only 6 consumed keys** (env.py:247-253):
`vehicle_mass_mult` U(0.85,1.15), `tire_friction_mult` U(0.80,1.20),
`surface_friction_mult` U(0.75,1.10), `sensor_noise_mult` U(0.5,2.0),
`spawn_lateral_jitter_m` U(−1,1), `spawn_heading_jitter_deg` N(0,5°).
Extra params are sampled but ignored. No actuator delay, no per-step noise
resampling, no obstacle randomization.

---

## 9. Curriculum

`sim_env/curriculum.py`: `CurriculumStage` {stage_id, scenario_id,
target_metric, advancement_threshold, min_episodes, environment_overrides}.
`sim_experiment/curriculum_runtime.py`: `CurriculumController` — advances only
when metric is present in **evaluation aggregate** AND `episodes_in_stage ≥
min_episodes` AND threshold met (≤ for lower-is-better metrics). Per-stage
seed derivation (`base + env + stage*10000`), audit history, state persisted
to `curriculum_state.json` and inside checkpoints; fingerprint mismatch blocks
resume. Default 5-stage curriculum exists but uses scenario overrides on one
track — **multi-track curricula don't exist yet**.

---

## 10. Vehicle Dynamics (`sim_core/`)

Dynamic single-track (bicycle) model, per-axle tire forces:
- tanh-saturating brush tire `Fy = −μFz·tanh(C·α/μFz)` with friction-ellipse
  coupling (longitudinal force consumes grip first);
- longitudinal load transfer only (no lateral); RWD default; brake bias 0.65;
- aero drag + rolling resistance; low-speed kinematic blend below 1.5 m/s;
- mass 1200 kg, top speed 45 m/s, max steer 0.58 rad, steering rate 4.5 rad/s;
- planar only (pitch/roll/z never integrated; banking/elevation stored but
  unused); no collision response; `reverse_max_speed` config dead.

Validated by `tests/test_vehicle_dynamics_validation.py` (determinism,
timestep independence via substepping) + `tools/vehicle_dynamics/` maneuvers.
Working tree has uncommitted vehicle-dynamics realism work
(`vehicle_model.py`, `vehicle_config.py`, `sensor_manager.py` modified).

**Sensors:** vehicle_state 60 Hz; lidar 15-ray/180°/40 m/30 Hz; camera
84×84/75°/30 Hz; IMU 100 Hz (accel+gyro, bias drift). All have Gaussian noise
+ latency-buffer support; rates independent.

---

## 11. Protocol & Communication (`sim_net/`, `sim_client/`)

NDJSON over TCP (`{"type","payload"}` lines), `TCP_NODELAY`, synchronous
request/response lockstep — sim steps only on STEP. Protocol 2.1
(`SET_SCENARIO` ≥2.1). Messages: HANDSHAKE(+ACK), DISCOVER_CONTRACT(+ACK —
full env contract incl. field classes & capabilities), RESET(+ACK), STEP(+ACK),
GET_STATE(+ACK, diagnostics only), SET_SCENARIO(+ACK), ERROR.

- `SimulationServer`: 1 client/1 env, `poll_and_process()` from host loop.
- `SimServerMulti`: one process/port, N envs, env-per-connection; envs pooled
  and recycled dirty (client must RESET).
- `SimulationClient`: connect/reset/step/set_scenario/get_state; no retry or
  reconnect; socket.timeout propagates.
- `SimGymEnv` (`sim_client/gym_env.py`): full gymnasium adapter; action space
  from server spec; obs = flat Box(23) or Dict{vector, image 84×84×3}.
- Headless: `python main.py --headless [--num-envs N] [--port P]`;
  `HeadlessSimProcessPool`/`HeadlessEnvPool` programmatic pools.

**Throughput:** ~350 SPS single-env; benchmarked state_only 1394 SPS vs full
sensor suite ~342; in-process vec envs show ~zero aggregate scaling
(332 SPS @4 envs — serial CPU stepping). `ProcessVectorEnv` (true subprocess
parallelism) is the right tool for scale-out on this CPU-only box.

---

## 12. Existing Trainers & Recorded Results

Runners in `sim_client/agents/` (all from-scratch PyTorch, CPU), wrapped by
`sim_experiment/trainers/*_trainer.py` (JSON contract-driven subprocess):

| Algo | Net | Resume | Notes |
|---|---|---|---|
| PPO (`ppo_baseline.py`) | actor+critic 2×128 tanh, diag Gaussian | yes (model+opt+step offset) | CleanRL-style; GAE; truncation bootstrap; vec-env; `act_dim` **hardcoded 3** |
| SAC (`sac_baseline.py`) | actor+twinQ 2×256 ReLU, auto-α | yes | 100k buffer, warmup 1k, τ .005 |
| DQN (`dqn_baseline.py`) | 2×128 ReLU | yes | discrete only; docstring says double-DQN but uses vanilla target-max |
| BC (`bc/`) | 64×64 MLP | yes | consumes `transitions_v1` datasets |
| `ppo_train.py` | — | — | **no-op skeleton**: collects but never updates — misleading name |

Checkpoint payload = `torch.save` dict (state_dicts + optimizers + dims +
seed + metrics + timestep + optional curriculum_state). Run/eval plumbing via
`sim_experiment` is mature and verified.

### Recorded learning results (all degenerate — the evidence)
- `experiments/baseline_ppo/metrics.json`: 49,152 steps, 54 eps, return
  719.3 flat (first ep already 720.8), **all lengths = 900 (timeout)**,
  completion 0, collisions 0, **mean speed ~0.01-0.05** (≈stationary),
  explained variance ≈0, entropy *rising*.
- `experiments/baseline_ui_env/metrics.json`: 10,240 steps, return 1582.6,
  same stationary pattern.
- `experiments/exported/exp_556fe0488d4db907`: 512-step smoke run; frozen
  evals identical across seeds (1185.85, len 1501, speed 0.0033) —
  deterministic near-zero action output.

**Conclusion: plumbing verified end-to-end; zero evidence of learned driving.
Reward shaping + progress incentives are the first-order problem.**

---

## 13. Datasets, Evaluation, Experiments

- `transitions_v1` datasets (`dataset.py`): episodes.jsonl + manifest,
  fingerprint-checked, deterministic split; strict agent/diagnostic isolation.
- Per-episode trajectory JSONL (`trajectory.py`) with sampling modes.
- `evaluation.py`: `evaluate_policy` → per-episode {reward, length,
  termination_reason, completed, collided, off_road, timed_out,
  mean_lateral/heading_error, mean_speed} + aggregate rates;
  `make_policy_from_checkpoint` adapters per algo.
- `metrics.py` append-only metrics.jsonl; `analytics.py` series/compare;
  `convergence.py` verdicts.
- Manifests fingerprinted (`experiment_id = exp_<sha[:16]>`); run dirs
  contain contract/metrics/checkpoints/eval/replays/trajectories/
  run_result; resume supported; batch scheduler + remote workers exist.
- `reproduce.py` honestly reports `exact_reproduction_guaranteed: False`.

---

## 14. Baselines Available

`pid_driver` (heuristic lane-follower on 23-dim obs — the strongest baseline,
actually drives), `random_agent`, plus fixed checkpoint eval. PID is the
correct "does RL beat a hand-tuned controller" comparison.

---

## 15. Gap Analysis vs. the Master Task

| Requirement | Status |
|---|---|
| Env/protocol/lifecycle | Mature |
| Observation contract + anti-leak | Exists; policy uses AGENT_OBSERVATION only |
| Reward decomposition | Built-in; needs rebalancing (stationary exploit) |
| Termination/truncation | Correct semantics; `custom_threshold` unimplemented |
| Action contract | Normalized + rate-limited env-side |
| Scenarios/obstacles | Deterministic obstacles only — need per-episode randomized placement (client-side via SET_SCENARIO) |
| Domain randomization | 6 params only; no obstacle/actuator-delay randomization |
| Curriculum | Runtime exists; needs multi-track extension |
| Vector envs | Sync + Process vec envs; TCP modes |
| Trainers | PPO/SAC/DQN exist but CPU-MLP only, PPO act_dim hardcoded, no image support, no temporal models |
| Checkpoints/resume | Verified |
| Evaluation | Single-env eval; needs multi-track/retention/generalization matrices |
| Multi-track/continual learning | **Absent entirely** — the core new work |
| Failure analysis | termination_reason exists; no classified failure DB |
| Vision pipeline | Camera obs exists; no CNN encoder in any trainer; fallback rasterizer low-fidelity headless |
| Temporal models | None (no frame stack, no recurrence) |
| Tests for agent lab | `tests/agent/` empty |
| Reward-hack resistance | Default config provably exploitable (all runs degenerate) |

## 16. Risks & Constraints

1. **CPU-only torch** — small MLPs, modest batch, ProcessVectorEnv for
   throughput; pixel-RL likely impractical headless (fallback rasterizer is
   low-fidelity anyway).
2. **~350 SPS per env serial** — real experiments need hours, not minutes;
   experiment matrix must be sized accordingly.
3. **Dual legacy/compiled pipelines** — use the declarative AgentDefinition
   path exclusively.
4. **PPO `act_dim` hardcoded 3** — fine for the default action space; wrap,
   don't duplicate.
5. **Step-after-done returns both flags true** — treat `terminated or
   truncated` as done; bootstrap value on truncation only.
6. **Simulator cleanliness** — envs recycled dirty in SimServerMulti; always
   RESET on connect.
7. **Uncommitted working-tree changes** — vehicle dynamics + UI files modified
   and new test/tool files untracked; `agentRL` work must not disturb them.
