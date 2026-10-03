# Glossary

Terms used across the simulator and this documentation.

## Simulation core

| Term | Meaning |
|---|---|
| **dt** | Fixed simulation timestep `1/60 s` — one `env.step()` advances exactly this far |
| **substep** | Internal physics integration step; `physics_substeps = 4` → 240 Hz effective |
| **single-track model** | "Bicycle" vehicle model — forces aggregated per axle rather than per wheel |
| **friction ellipse** | Tire model where longitudinal force consumes lateral grip capacity |
| **kinematic blend** | Low-speed (< 1.5 m/s) fallback to kinematic bicycle steering response |
| **OBB2D** | Oriented bounding box — rectangle with position + yaw used for collision |
| **SAT** | Separating Axis Theorem — the OBB↔OBB collision test |
| **broadphase** | Spatial-hash pre-filter (12 m cells) that culls collision candidates |

## Track & world

| Term | Meaning |
|---|---|
| **control point (CP)** | Spline node: position + road width + elevation/banking/friction |
| **spline** | Catmull-Rom curve through control points, arc-length parameterized |
| **s** | Arc-length position along the track centerline |
| **boundary** | Road-edge wall: `guardrail · wall · curb · open · invisible` |
| **checkpoint** | Cross-track progress gate; `num_checkpoints` along the track |
| **spawn** | Vehicle start pose (`pos`, `yaw`, `initial_speed`) + jitter |
| **entity** | World object (obstacle, barrier, cone, sign, traffic light, …) |
| **`.sim.json`** | The serialized project document containing the whole environment |

## Environment contract

| Term | Meaning |
|---|---|
| **environment** | The `Simulator = World + Physics + Sensors + Rewards + …` the agent acts in |
| **agent** | External policy — your model. The simulator never owns the agent's brain |
| **observation** | Data the agent may see — flat `Box(23)` by default |
| **action** | `[steer, throttle, brake]` continuous or discrete options |
| **reward component** | One weighted reward term (`progress`, `centering`, …) |
| **reward breakdown** | Per-component contributions in `info["reward_breakdown"]` |
| **termination** | Episode-ending condition (`terminated`) vs truncation (`truncated`) |
| **scenario** | Runtime modifier: weather, friction, spawn, target speed, noise |
| **randomization (DR)** | Domain randomization — distribution over env parameters per reset |
| **oracle / leakage guard** | Channels categorized non-`agent_observation` that policies may not see |
| **fingerprint** | SHA-256 over canonical project JSON — binds runs to an exact environment |

## Studio & protocol

| Term | Meaning |
|---|---|
| **Studio** | The interactive app (`python main.py` — home + workspace tabs) |
| **EDIT / SIMULATE / REPLAY / DATA / DYNAMICS** | The five workspace tabs |
| **Environment Inspector** | Right panel — GEO + RL tab groups editing the project |
| **headless** | `main.py --headless` — physics + TCP server, no window |
| **multi-env** | `--num-envs N` — one port, N env slots assigned FIFO |
| **NDJSON** | Newline-delimited JSON — the TCP wire framing |
| **lockstep** | The env advances exactly once per `STEP` message — no free-running sim |
| **SimGymEnv** | `gymnasium.Env` adapter wrapping `SimulationClient` |

## Experiments & data

| Term | Meaning |
|---|---|
| **manifest** | Immutable experiment definition (`experiment.json`) |
| **run** | One execution of a manifest — own dir, state, metrics |
| **env_mode** | How envs are hosted: `inprocess · process · tcp · tcp_multi` |
| **trajectory** | Per-episode JSONL of agent_data + diagnostic_data during training |
| **`transitions_v1`** | Exported offline dataset format (episodes + manifest) |
| **schema_hash** | SHA-256 over obs+action schema — matches datasets to environments |
| **curriculum** | Ordered scenario stages with metric-based advancement |
| **episode recording** | Studio-recorded telemetry+action replay (no observations) |

## See also

- [FAQ](faq.md) · [Compatibility](compatibility.md)
- [Architecture](../architecture/architecture.md)
