# Project Schema — `.sim.json`

Every environment is one JSON document (`*.sim.json`, `indent=2`) —
track geometry, entities, vehicle, and the complete RL contract in a
single file. The studio's undo system snapshots the whole document, so
geometry edits and reward edits share one history.

- `schema_version`: `"2.0.0"` written today (`"3.0.0"` recognized;
  `"1.0.0"` and older are migrated on load).
- `environment_version`: semantic project version (`"1.0.0"` initially);
  the patch digit auto-increments on save when content changed.
- `fingerprint`: SHA-256 over the canonical JSON, excluding metadata keys
  (`version`, `timestamp`, `last_saved`, `author`,
  `environment_version`, `schema_version`, `fingerprint`). Used to detect
  "did anything actually change" and to bind experiments to an exact
  environment.

## Top-level structure

```jsonc
{
  "schema_version": "2.0.0",
  "environment_version": "1.0.0",
  "name": "Proving Ground Oval",
  "road_definition":    { ... },
  "vehicle_config":     { ... },
  "action_config":      { ... },   // legacy action block
  "observation_schema": { ... },   // legacy observation block
  "reward_config":      { ... },   // legacy reward block
  "termination_config": { ... },   // legacy termination block
  "randomization_config": { ... }, // legacy randomization block
  "scenario_config":    { ... },   // legacy scenario block
  "entities":           [ ... ],
  "agent":              { ... },   // declarative agent — takes precedence
  "episode_config":     { ... },
  "scenario_def":       { ... },
  "curriculum":         { ... },   // optional
  "experiment_config":  { ... },   // optional
  "fingerprint":        "sha256…"
}
```

### Two config paths — important

The file carries **legacy flat blocks** (`action_config`,
`observation_schema`, `reward_config`, `termination_config`,
`randomization_config`, `scenario_config`) *and* a **declarative agent
block** (`agent.observation_space`, `agent.action_space`,
`agent.reward_function`, `agent.termination_rules`,
`scenario_def.randomization`).

> **When `agent` is present, the compiled declarative pipelines take
> precedence** and the legacy blocks are bypassed at runtime. Both are
> serialized for compatibility. The inspector edits the agent block.

---

## `road_definition`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `"Default Track"` | Track display name |
| `is_closed` | bool | `true` | Closed loop vs open route |
| `control_points` | list | — | Ordered spline control points (≥ 3) |
| `boundary_config` | obj | — | Left/right boundary style |
| `spawn_point` | obj | — | Start pose |
| `num_checkpoints` | int | `16` | Progress gates (min 4 effective) |
| `default_friction` | float | `1.0` | Base surface µ multiplier |

`control_points[i]`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `x`, `y` | float | required | Plan position, meters |
| `z` | float | `0.0` | Elevation, meters |
| `width` | float | `12.0` | Full road width at the point |
| `banking` | float | `0.0` | Serialized; **not consumed by dynamics** |
| `friction` | float | `1.0` | Serialized; **not consumed** — use `default_friction` / scenario |

`boundary_config`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `left_type`, `right_type` | enum | `"guardrail"` | `guardrail` · `wall` · `curb` · `open` · `invisible`. Only guardrail/wall render barrier mesh — **all types still produce collision segments at the road edge** |
| `wall_height` | float | `0.8` | Barrier visual height, m |
| `curb_width` | float | `0.5` | Curb strip outside road edge, m |
| `curb_height` | float | `0.15` | Curb raise, m |
| `has_curbs` | bool | `true` | Curbs rendered + boundary pushed out |
| `has_lane_markings` | bool | `true` | Serialized; not used by mesh gen |

`spawn_point`: `{x, y, z = 0.2, yaw (rad), initial_speed}`.

---

## `vehicle_config`

Full table in [Vehicle Dynamics](../vehicle-dynamics/vehicle-model.md#parameters-vehicle_config-in-simjson)
— all 27 fields (`mass` … `low_speed_relax_tau`) with defaults.

## `action_config` (legacy) and `agent.action_space` (declarative)

Legacy block:

| Field | Type | Default |
|---|---|---|
| `type` | `"continuous" \| "discrete"` | `"continuous"` |
| `continuous_low` | `[steer, throttle, brake]` | `[-1.0, 0.0, 0.0]` |
| `continuous_high` | same order | `[1.0, 1.0, 1.0]` |
| `discrete_actions` | list of `[steer, thr, brk]` | 5 options: Coast `[0,0,0]`, Accelerate `[0,0.7,0]`, Brake `[0,0,0.8]`, Left `[−0.6,0.4,0]`, Right `[0.6,0.4,0]` |

`agent.action_space`:

| Field | Type | Meaning |
|---|---|---|
| `space_type` | `"continuous" \| "discrete"` | |
| `channels` | list | Per-channel config (continuous) |
| `discrete_options` | list | `{name, values[]}` |

Channel fields: `name`, `channel_type`, `min_val`/`max_val`,
`default_val`, `scaling`, `dead_zone` (|deviation from default| < dz →
default), `rate_limit` (units/s, 0 = unlimited), `description`.
Defaults: steering `[−1,1]` dz 0.02 rl 4.0/s · throttle `[0,1]` dz 0.01
rl 5.0/s · brake `[0,1]` dz 0.01 rl 6.0/s.

## `observation_schema` (legacy) and `agent.observation_space` (declarative)

Legacy block — `include_*` booleans + `flatten_vector`:

| Field | Default | Channel produced |
|---|---|---|
| `include_speed` | `true` | speed ÷ 45 |
| `include_velocity` | `true` | vel_body [vx ÷ 45, vy ÷ 10] |
| `include_yaw_rate` | `true` | yaw_rate ÷ 3 |
| `include_steering_angle` | `true` | steering ÷ 0.6 |
| `include_distance_from_center` | `true` | lateral offset ÷ half-width |
| `include_heading_error` | `true` | ÷ π |
| `include_distance_to_checkpoint` | `true` | ÷ 100 |
| `include_lidar_rays` | `true` | `ranges_norm` [15] |
| `include_camera_rgb` | `false` | 84×84×3 image (Dict obs) |
| `flatten_vector` | `true` | flat Box(23) vs Dict |

`agent.observation_space`:

| Field | Type | Meaning |
|---|---|---|
| `channels` | list | `ObservationChannelConfig` per feature |
| `flatten_vector` | bool | flat vector vs per-channel Dict |
| `include_image_channel` | bool | legacy single-image flag |
| `image_channels` | `[{name, shape}]` | authoritative multi-camera contract |

Channel fields: `name`, `channel_type` (`scalar`/`vector`/`image`),
`shape`, `dtype` (`float32`/`uint8`), `range_low`/`range_high`,
`normalization` (`none`/`scale`/`min_max`/`clip`; `standardized`
declared but not implemented), `norm_params`, `source_sensor`,
`source_key`, `category` (`agent_observation` / `debug_telemetry` /
`oracle_ground_truth` — the latter two trigger the leakage guard),
`enabled`, `description`.

## `reward_config` (legacy) and `agent.reward_function` (declarative)

Legacy weights (all ≥ 0):

| Field | Default | Meaning |
|---|---|---|
| `weight_progress` | `1.0` | per-meter progress |
| `weight_centering` | `0.5` | centerline reward |
| `weight_speed` | `0.2` | fraction of `target_speed` |
| `target_speed` | `20.0` | m/s |
| `weight_heading` | `0.3` | `cos(heading_error)` |
| `weight_action_smoothness` | `0.05` | `−(Δsteer)²` |
| `checkpoint_bonus` | `10.0` | per gate |
| `lap_completion_bonus` | `100.0` | per lap |
| `collision_penalty` | `50.0` | on collision |
| `off_road_penalty` | `25.0` | off road |
| `backward_penalty` | `1.0` | reverse driving |

`agent.reward_function.components[]`: `{component_id, name,
component_type, weight, enabled, params, description}` — the default
11-component table is in [RL Overview](../reinforcement-learning/rl-overview.md#reward).

## `termination_config` (legacy) and `agent.termination_rules` (declarative)

Legacy flags:

| Field | Default | Meaning |
|---|---|---|
| `terminate_on_collision` | `true` | |
| `terminate_on_off_road` | `true` | |
| `terminate_on_wrong_direction` | `true` | |
| `wrong_direction_max_angle_deg` | `120.0` | heading-error threshold |
| `terminate_on_lap_completion` | `false` | |
| `laps_to_complete` | `1` | |
| `max_episode_steps` | `5000` | truncation |
| `max_seconds_without_checkpoint` | `30.0` | truncation (stuck) |

`agent.termination_rules.rules[]`: `{rule_id, name, condition_type,
enabled, is_truncation, params, description}` — condition types
`collision · off_road · wrong_direction · course_completion · max_steps ·
simulation_timeout · checkpoint_timeout` (`custom_threshold` declared but
not evaluated).

## `randomization_config` (legacy)

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | |
| `mass_range` | `[0.85, 1.15]` | uniform multiplier |
| `tire_friction_range` | `[0.80, 1.20]` | |
| `surface_friction_range` | `[0.75, 1.10]` | |
| `sensor_noise_multiplier_range` | `[0.5, 2.0]` | |
| `spawn_lateral_jitter_m` | `1.0` | uniform ± |
| `spawn_heading_jitter_deg` | `10.0` | uniform ± |

Declarative `scenario_def.randomization`: `{enabled, global_seed,
parameters{<name>: {param_name, distribution: fixed|uniform|normal,
param1, param2, clip_min, clip_max}}}` — param1 = fixed value / uniform
min / normal mean; param2 = uniform max / normal std.

## `scenario_config` (legacy) and `scenario_def` (declarative)

Legacy: `{name, time_of_day: day|dusk|night, weather: clear|rain|fog,
ambient_light (0.1–1.0), surface_friction_mult, obstacles[],
difficulty_level}`.

`scenario_def`: `{scenario_id, name, description, weather, time_of_day,
ambient_light, surface_friction_mult, target_speed_override,
time_limit_override, sensor_noise_mult, spawn_override{pos, yaw_deg,
initial_speed}, obstacle_overrides[{entity_type, pos, yaw, name}],
randomization}`.

## `entities[]`

Per-class fields (see [Track Editor](../track-editor/track-editor.md#environment-entities)):
`{entity_id, name, entity_type, semantic_label, pos{x,y,z}, yaw,
is_collidable}` + type params — see
[entities](../track-editor/track-editor.md#environment-entities).

## `episode_config`

| Field | Default | Status |
|---|---|---|
| `max_duration_seconds` | `60.0` | enforced (env-level truncation) |
| `max_steps` | `3600` | **serialized, unused** — use termination rules |
| `spawn_mode` | `"track_spawn"` | only `"custom_pose"` implemented; `"random_cp"` ignored |
| `initial_speed` | `0.0` | |
| `custom_spawn_pos` / `custom_spawn_yaw_deg` | `(0,0,0.2)` / `0` | for `custom_pose` |
| `spawn_lateral_jitter_m` / `spawn_heading_jitter_deg` | `0.0` | |
| `random_seed` | `42` | default reset seed |
| `reset_behavior` | `"hard"` | **config only** |
| `auto_reset_on_done` | `false` | **unused** |

## `agent`

`{agent_id, name, entity_type ("vehicle"), entity_id, sensor_configs[],
sensor_names[], observation_space, action_space, reward_function,
termination_rules, spawn_config{spawn_mode,pos,yaw_deg,initial_speed,
lateral_jitter_m,heading_jitter_deg}}`.

`sensor_configs[i]`: `{name, sensor_type (camera_rgb|lidar_rays|imu|
vehicle_state), enabled, params{...}}` — the authoritative sensor list;
`sensor_names` is the derived flat list kept for back-compat. Per-type
params: [Sensors](../sensors/sensors.md).

`spawn_config` fields are serialized; only `initial_speed` and
`lateral_jitter_m` are currently editable via the inspector AGENT tab.

## Optional blocks

- `curriculum` — stage list (scenario_id, target_metric,
  advancement_threshold, min_episodes, environment_overrides).
- `experiment_config` — default experiment scaffold (algorithm,
  timesteps, hyperparameters, eval/checkpoint cadence).

## Load behavior

- Missing `schema_version` → treated as `"1.0.0"`, migrated
  (`obstacles`/`scenario_config.obstacles` → `entities`, missing `agent`
  → default vehicle agent, missing `episode_config`/`scenario_def`
  synthesized).
- Unknown keys are ignored; missing keys get defaults — **tolerant
  loading**. Validation is a separate report (VALIDATE tab /
  `validate-env` CLI), not a load gate.
- On save, `environment_version` patch bumps only when the fingerprint
  differs from the last save.

## See also

- [Track Editor](../track-editor/track-editor.md) — authoring the same data visually
- [Settings Reference](settings-reference.md) — studio-level settings
- [RL Overview](../reinforcement-learning/rl-overview.md) — what these blocks compile into
