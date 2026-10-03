# Compatibility & Version Map

Version numbers in this codebase and which ones inter-operate.

## Versions at a glance

| Layer | Current | Constant |
|---|---|---|
| Simulator | `4.0.0` | `sim_version.py` → `VERSION` |
| TCP protocol | `2.1` (accepts `2.0`) | `sim_net/protocol.py` |
| Project schema | `2.0.0` written; `3.0.0`, `1.x` readable | `sim_project/schema.py` |
| Dataset format | `transitions_v1` | `sim_experiment/datasets.py` |
| Trajectory format | `trajectory_jsonl_v1` | `sim_experiment/trajectories.py` |
| Contract version | `1.0.0` | `sim_env/versioning.py` |

## Protocol compatibility

| Client offers | Server response | Features |
|---|---|---|
| `["2.1"]` | `2.1` | Full — incl. `SET_SCENARIO` |
| `["2.0"]` | `2.0` | All except `SET_SCENARIO` (returns `unsupported_in_protocol_version`) |
| `[]` / missing `protocol_versions` | `2.1` | Server assumes the current version |
| no mutual version | `protocol_version_mismatch` | Connection rejected |

Legacy `client_protocol_versions` in HANDSHAKE is still accepted.

## Schema compatibility

- **`schema_version` missing** → treated as `"1.0.0"`, migrated on load
  (old `obstacles[]` → `entities[]`, `scenario_config.obstacles` →
  entities, missing `agent` → default vehicle agent, `episode_config` /
  `scenario_def` synthesized).
- **Unknown keys ignored**; missing keys get defaults — tolerant loading.
- `environment_version` patch bumps automatically on save when the
  fingerprint changes.

## Dataset compatibility

- A `transitions_v1` dataset carries `schema_hash` (SHA-256 of
  obs+action schema) and `env_fingerprint` — `train-bc` requires both to
  match the target environment.
- `dataset-validate` reports the exact incompatibility in `errors[]`.

## Experiment compatibility

- Manifest stores `environment_fingerprint` + `scenario_id` — `launch`
  re-validates the project path still produces that fingerprint.
- `reproduce` rebuilds the manifest and re-verifies fingerprints.
- `export`/`archive` bundle run + manifest + checkpoint artifacts.

## Declared-but-unimplemented fields

Honesty list — fields present in serialization that do **nothing** today:

| Field | Where | Reality |
|---|---|---|
| `reverse_max_speed` | `vehicle_config` | unused |
| per-point `friction`, `banking` | `road_definition.control_points` | carried through spline; not consumed by dynamics |
| `spawn_mode "random_cp"` | `episode_config` | ignored — only `custom_pose` implemented |
| `auto_reset_on_done`, `reset_behavior` | `episode_config` | unused |
| `include_position`, `include_laps`, `include_progress`, `include_obstacle_distances` | legacy `observation_schema` | parsed but produce nothing (position is always in obs anyway) |
| `normalization "standardized"` | observation channels | declared, not implemented |
| `CUSTOM_THRESHOLD` | termination conditions | declared, not evaluated |
| `sim_step_count` | `GET_STATE` response | always `0` |
| `pitch`, `roll` | vehicle state | never integrated — always 0 |
| `has_lane_markings` | `boundary_config` | serialized only |
| `spawn_config.spawn_mode` | `agent.spawn_config` | plumbing exists, no tab effect |

## Runtime compatibility

| Dependency | Tested version | Required for |
|---|---|---|
| Python | 3.13.7 | all |
| numpy | 2.2.6 | all |
| pygame | 2.6.1 | studio |
| moderngl | 5.12.0 | studio + camera sensor |
| opencv-python | 4.x | camera sensor |
| gymnasium | 1.2.1 | `SimGymEnv` + trainers |
| torch | 2.10.0 | PPO/SAC/DQN/BC trainers |
| pytest | 9.1.0 | tests |
| pyinstaller | 6.22.3 | standalone build |

The standalone bundle excludes torch & friends — training requires a
source install.

## See also

- [Glossary](glossary.md) · [FAQ](faq.md)
- [TCP Protocol](../agents/tcp-protocol.md) · [Project Schema](../configuration/project-schema.md)
