# Sensors

Complete reference for the sensor suite attached to the simulated vehicle — the four
serialized sensor types, their outputs, noise and latency models, and how sensor
samples flow into the agent's observation channels.

Sources: `sim_core/sensors/` (runtime), `sim_env/sensor_config.py` (serialization),
`sim_ui/inspector.py` (UI editing).

## Sensor suite architecture

Sensors are declared per-agent. `agent.sensor_configs` (a list of `SensorConfig`
objects: `name`, `sensor_type`, `enabled`, `params{}`) is the **source of truth**
and is what gets serialized into the `.sim.json` project file. `agent.sensor_names`
is a derived list of *enabled* sensor names kept only for back-compat with older
consumers and validation.

At environment build time, `SensorManager.build_from_configs()` calls
`SensorConfig.build_sensor()` for each enabled config. Sensor priority in
`SimulationEnvironment.__init__`:

1. An explicitly injected `sensor_manager` argument.
2. `agent.sensor_configs` (declarative path — normal for UI-authored projects).
3. `SensorManager.create_default_sensor_suite()` (4 sensors — see below).

`SensorManager.update_all(sim_time, context, rng)` returns `{name: sample}` and
honors each sensor's own update rate. All sensors draw from the environment's
shared `np.random.default_rng(seed)` (owned by `FixedClock`), so noise and bias
drift are deterministic for a given seed.

The default suite (`create_default_sensor_suite()` / `default_suite_configs()`):

| Name | Type | Rate |
|---|---|---|
| `vehicle_state` | `vehicle_state` | 60 Hz |
| `lidar_rays` | `lidar_rays` | 15 rays / 180° / 40 m — 30 Hz |
| `rgb_camera` | `camera_rgb` | 84×84 / 75° — 30 Hz |
| `imu` | `imu` | 60 Hz (runtime suite) / 100 Hz (serialized defaults) |

## Shared base-class behavior (`BaseSensor`)

Every sensor inherits these parameters (`sim_core/sensors/base_sensor.py`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | str | per type | Unique key in the samples dict; used by observation channels |
| `update_frequency_hz` | float | 30.0 | Sample rate, clamped to ≥ 1 Hz |
| `local_pos` | Vec3 | (0, 0, 1.2) | Mount offset in the vehicle frame (+x fwd, +y left, +z up) |
| `local_yaw` | float | 0.0 | Mount yaw offset relative to vehicle heading (rad) |
| `noise_std` | float | 0.0 | Gaussian noise σ (units depend on sensor) |
| `latency_seconds` | float | 0.0 | Output delay simulated via a timestamped history buffer |

Behavior, per `update(sim_time, context, rng)`:

- **Rate gating** — the sensor only re-samples when
  `sim_time - last_update >= 1/f`. Between updates the **stale sample is held**
  and returned unchanged.
- **Latency** — when `latency_seconds > 0`, new samples are pushed onto a history
  buffer and the latest sample with `t ≤ sim_time − latency` is returned (the
  freshest sample is returned until the buffer fills).
- **Noise** — `N(0, noise_std)` applied elementwise to the raw sample before it
  enters the latency buffer.
- **Context** — `_generate_raw_sample(context, rng)` receives
  `context['vehicle']`, `context['track_queries']`, `context['checkpoint_idx']`,
  and `context['obstacle_segments']`.
- `reset()` clears the last sample and history (called by
  `SensorManager.reset_all()` on episode reset). The IMU additionally clears its
  bias accumulators.

Vehicle-mount convention: `local_pos.x` is forward, `local_pos.y` is left,
`local_pos.z` is up, in meters; `local_yaw` rotates the sensor's forward axis
relative to the vehicle's heading.

---

## `camera_rgb` — RGB camera (`CameraSensor`)

**Purpose.** Forward-facing color camera for visual RL / perception. Produces a
uint8 RGB raster each update.

**Output.** `np.ndarray`, dtype `uint8`, shape `(height, width, 3)` — e.g.
`(84, 84, 3)`. RGB channel order.

**Frame.** Mounted at `local_pos` in the vehicle frame; looks along vehicle
forward + `local_yaw`, pitched by `local_pitch`.

### Parameters

| Name | Type | Default | Range / notes |
|---|---|---|---|
| `width` | int | 84 | inspector 32–512 px (step 16) |
| `height` | int | 84 | inspector 32–512 px (step 16) |
| `fov_degrees` | float | 75.0 | inspector 30–120° (step 5) |
| `update_frequency_hz` | float | 30.0 | inspector 1–120 Hz |
| `local_pos` | [x,y,z] | [1.0, 0.0, 1.1] | hood/roof mount; inspector ±5 m per axis |
| `local_yaw` | float | 0.0 | inspector ±180° |
| `local_pitch` | float | −0.05 rad | slight downward tilt; inspector ±90° (displayed in degrees, stored rad) |
| `noise_std` | float | 0.0 | inspector 0–0.5 (row), clamped 0–1 |
| `latency_seconds` | float | 0.0 | inspector 0–1 s |

### Image generation — two paths

1. **Real offscreen render** — if a renderer callable has been injected via
   `set_offscreen_renderer()` (done by the 3D rendering engine in the interactive
   studio) and it returns an image of shape `(h, w, 3)`, that real 3D frame is
   used.
2. **Procedural fallback** (`_synthesize_perspective_road`) — otherwise a
   synthetic 2D perspective sketch is rasterized with OpenCV: sky-blue gradient
   over the top 45 % of the frame, green ground, a dark-gray road trapezoid
   whose center is shifted by `lateral_offset · (w/18)` and
   `heading_error · (w/2.5)`, red curbs, and a dashed white centerline. This is
   **not** a real 3D render — it contains no entities, barriers, or track
   geometry beyond an idealized straight road, but it works headless and is
   consistent enough for RL.

**Caveat:** `fov_degrees` and `local_pitch` are stored and serialized but are
**unused by the procedural fallback path** — they only matter when an offscreen
renderer is attached.

### Noise model

`N(0, noise_std · 255)` added to every pixel, clipped to `[0, 255]`, cast back to
`uint8`. `noise_std = 0.1` therefore means σ ≈ 25.5 intensity levels.

### Limitations

- Procedural path ignores `fov_degrees` / `local_pitch` and shows no obstacles or
  boundary geometry — it encodes only lateral offset and heading error.
- `local_pos`/`local_yaw` do not affect the procedural image at all (it is always
  a forward view).
- 84×84 is small; raise `width`/`height` for tasks that need finer detail.

### RL usage

Cameras are **not** referenced through `source_sensor`/`source_key` vector
channels. Instead the observation space carries `image_channels`:
`[{"name": <sensor_name>, "shape": [H, W, 3]}]`. At runtime the compiled
pipeline emits `obs["image"]` when the camera is named exactly `rgb_camera`,
otherwise `obs["image_<sensor_name>"]`, alongside `obs["vector"]`. Missing
camera data yields a zeroed `uint8` image.

### Editing in the inspector

SENSORS tab → `EDIT` on a camera exposes Update Rate (1–120 Hz), FOV (30–120°),
Width/Height (32–512), Mount X/Y/Z (±5 m), Mount Yaw (±180°), Mount Pitch
(±90°), Pixel Noise, Latency, and an **"In Observations"** toggle that adds or
removes the camera from `observation_space.image_channels` (shape is synced from
width/height automatically).

---

## `lidar_rays` — planar LiDAR (`RaycastSensor`)

**Purpose.** 2D radial rangefinder against track boundary segments and world
obstacle segments. The runtime class reports `sensor_type="raycast_lidar"`, but
the **serialized** type name is `lidar_rays`.

**Output.** `dict` per sample:

| Key | Type | Shape | Description |
|---|---|---|---|
| `distances` | float32 | `[num_rays]` | Range per ray in meters, clamped `[0, max_range]` |
| `ranges_norm` | float32 | `[num_rays]` | `distances / max_range` — 1.0 = clear, 0.0 = contact |
| `num_rays` | int | — | Beam count |
| `max_range` | float | — | Sensor max range (m) |

A ray that hits nothing returns `max_range` (miss = full range). Hit geometry
per ray is also cached in `sensor.ray_visuals` for the editor/3D rendering of
ray fans.

**Frame.** Planar (2D). Rays originate at `local_pos` transformed by the vehicle
pose and fan evenly from `−fov/2` to `+fov/2` around `vehicle_yaw + local_yaw`.
With `num_rays = 1` a single forward ray is cast. `local_pos.z` does not affect
ray casting (rays are horizontal).

### Parameters

| Name | Type | Default | Range / notes |
|---|---|---|---|
| `num_rays` | int | 15 | ctor clamps ≥ 1; inspector 3–64 |
| `fov_degrees` | float | 180.0 | inspector 30–360° (step 10); handler clamp 10–360 |
| `max_range` | float | 40.0 m | inspector 5–200 m |
| `update_frequency_hz` | float | 30.0 | inspector 1–120 Hz |
| `local_pos` | [x,y,z] | [0, 0, 1.2] serialized | runtime ctor default is [1.5, 0, 0.5] — see caveat below |
| `local_yaw` | float | 0.0 | inspector ±180° |
| `noise_std` | float | 0.0 | meters; inspector 0–0.5 (row), clamped 0–1 |
| `latency_seconds` | float | 0.0 | supported via base class; **no inspector row** for LiDAR |

### Hit model

Each ray calls `track_queries.cast_ray(origin, dir, max_range,
additional_segments)` against `track.all_boundary_segments` **plus**
`context['obstacle_segments']` (4 edges per collidable world entity). The nearest
intersection wins; on a miss the distance is `max_range` and `hit_point` is
`None`.

**Performance note:** ray casting is a brute-force linear scan over all boundary
+ obstacle segments (with an AABB prefilter per segment) — there is **no
broadphase**. Cost per step ≈ `num_rays × segment_count`; keep `num_rays` and
`max_range` modest on large tracks.

### Noise model

`N(0, noise_std)` (meters) added per beam, distances clipped to
`[0, max_range]`, then `ranges_norm` is recomputed from the noisy distances so
the two arrays stay consistent.

### Limitations

- 2D only — no vertical channel, no returns from curbs raised geometry or walls
  above ground plane (collision segments live at road edge).
- `latency_seconds` works at runtime but is not editable in the inspector for
  LiDAR (edit the `.sim.json` `params` block directly).
- Linear scan — see performance note above.

### RL usage

Vector channel `lidar_ranges` in the default observation space reads
`source_sensor="lidar_rays"`, `source_key="ranges_norm"` and feeds the 15
normalized ranges (raw, in `[0,1]`) directly into the flattened observation. If
the lidar is absent the pipeline emits `ones(15)` for that channel.

### Editing in the inspector

SENSORS tab → `EDIT` on a LiDAR exposes Update Rate, Beams (3–64), FOV
(30–360°), Range (5–200 m), Mount X/Y/Z, Mount Yaw, and Range Noise.

---

## `imu` — inertial measurement unit (`IMUSensor`)

**Purpose.** 6-axis accelerometer + gyroscope reporting **proper acceleration**
(specific force) in the body frame plus gravity on Z — matching the convention
`accel_body` uses in `VehicleState`.

**Output.** `dict` per sample:

| Key | Type | Shape | Description |
|---|---|---|---|
| `linear_acceleration` | float32 | `[3]` | `[ax, ay, 9.81]` — body proper accel (m/s²) + gravity on Z |
| `angular_velocity` | float32 | `[3]` | `[0, 0, yaw_rate]` — roll & pitch rates are **hardcoded 0** |

Because the vehicle model never integrates pitch or roll, the IMU's roll and
pitch gyro channels always read zero (plus noise/bias). This is a documented
limitation of the underlying single-track model, not a bug.

### Parameters

| Name | Type | Default | Range / notes |
|---|---|---|---|
| `update_frequency_hz` | float | 100.0 | class + serialized default; runtime default suite uses 60 Hz; inspector 1–120 |
| `accel_noise_std` | float | 0.05 | σ m/s² per axis; inspector 0–1 |
| `gyro_noise_std` | float | 0.01 | σ rad/s per axis; inspector 0–1 |
| `bias_drift_rate` | float | 0.001 | random-walk σ per sample; inspector 0–0.1 |
| `local_pos` | [x,y,z] | [0, 0, 0.5] serialized | runtime ctor default [0, 0, 0.3]; **not editable in inspector** |

The IMU constructor does **not** take `noise_std`/`latency_seconds` — the base
noise/latency machinery is present but not exposed for this sensor type.

### Noise & bias model

- Per-axis accelerometer noise `N(0, accel_noise_std)`, gyro noise
  `N(0, gyro_noise_std)` applied every sample.
- Random-walk bias: `accel_bias += N(0, bias_drift_rate)` and likewise for the
  gyro, accumulated **per sample** and added on top of noise. Bias resets on
  `reset()`. At 60 Hz a `bias_drift_rate` of 0.001 gives a slow wander with σ
  growing as √n — enough to matter for state-estimation experiments.

### Limitations

- Roll/pitch rates always 0 (single-track model has no suspension or roll
  dynamics).
- No latency/base-noise parameters; no inspector mount controls.

### RL usage

Feed any keys via vector channels, e.g. `source_sensor="imu"`,
`source_key="linear_acceleration"` (a `[3]` vector channel — set `shape: [3]`).

### Editing in the inspector

SENSORS tab → `EDIT` on an IMU exposes Update Rate, Accel Noise, Gyro Noise, and
Bias Drift only.

---

## `vehicle_state` — telemetry sensor (`VehicleStateSensor`)

**Purpose.** Ground-truth vehicle kinematics and track-relative telemetry — the
primary observation source for most agents.

**Output.** `dict` of floats (16 keys), body-frame velocities/accelerations plus
track-relative quantities from `track_queries.query_vehicle_pose()`:

| Key | Unit | Description |
|---|---|---|
| `speed` | m/s | Longitudinal body velocity `vel_body.x` |
| `vel_x` | m/s | Body-frame forward velocity |
| `vel_y` | m/s | Body-frame lateral velocity |
| `yaw` | rad | Heading (normalized) |
| `yaw_rate` | rad/s | Yaw rate `wz` |
| `accel_x` | m/s² | Proper acceleration, body x |
| `accel_y` | m/s² | Proper acceleration, body y |
| `steering_angle` | rad | Road-wheel angle (+ = left, since cmd −1 = left is mapped to +wheel) |
| `throttle` | [0,1] | Applied throttle |
| `brake` | [0,1] | Applied brake |
| `distance_from_center` | m | Lateral offset from centerline; **+ = left** |
| `heading_error` | rad | `normalize_angle(yaw − road_tangent)` |
| `is_on_road` | 0/1 | `|lat_offset| ≤ road_width/2` |
| `distance_to_checkpoint` | m | Euclidean distance to the current gate |
| `track_progress_s` | m | Arc-length position along the centerline |
| `road_width` | m | Local road width |

### Parameters

| Name | Type | Default | Range / notes |
|---|---|---|---|
| `update_frequency_hz` | float | 60.0 | matches env step rate; inspector 1–120 |
| `noise_std` | float | 0.0 | applied per-key; inspector 0–1 |
| `latency_seconds` | float | 0.0 | inspector 0–1 s |

### Noise model

`N(0, noise_std)` added to **every key except `is_on_road`** (which stays
exactly 0.0/1.0). Note the same σ applies to every field regardless of unit —
meters, radians, and normalized controls alike — so keep `noise_std` small.

### Limitations

- `distance_to_checkpoint` targets `checkpoints[checkpoint_idx % n]` — a straight
  -line distance, not along-track.
- Noise is uniform σ across heterogeneous units (see above).

### RL usage

Feeds most default observation channels (`speed`, `velocity_body`,
`steering_angle`, `distance_from_center`, `heading_error`,
`distance_to_checkpoint`) via `source_sensor="vehicle_state"` +
`source_key=<key>`.

### Editing in the inspector

SENSORS tab → `EDIT` exposes Update Rate, Noise, and Latency.

---

## How sensors feed observations

Each observation channel declares `source_sensor` + `source_key`. The compiled
pipeline indexes `samples[source_sensor][source_key]` every step, applies the
channel's normalization, and writes into the flattened vector (or dict when
`flatten_vector=False`).

| Channel (default space) | `source_sensor` | `source_key` | Normalization |
|---|---|---|---|
| `speed` | `vehicle_state` | `speed` | ÷ 45 |
| `velocity_body` | `vehicle_state` | `vel_x`, `vel_y` | ÷ 45, ÷ 10 |
| `yaw_rate` | `vehicle_state` | `yaw_rate` | ÷ 3 |
| `steering_angle` | `vehicle_state` | `steering_angle` | ÷ 0.6 |
| `distance_from_center` | `vehicle_state` | `distance_from_center` | ÷ half road width |
| `heading_error` | `vehicle_state` | `heading_error` | ÷ π |
| `distance_to_checkpoint` | `vehicle_state` | `distance_to_checkpoint` | ÷ 100 |
| `lidar_ranges` | `lidar_rays` | `ranges_norm` | none (already [0,1]) |

Cameras bypass `source_key` and attach through `image_channels` (see
`camera_rgb` above). Validation (`EnvironmentValidator`) raises an **ERROR** if a
channel references a `source_sensor` that is not attached to the agent, or if an
image channel references a disabled camera.

### Multiple cameras

Use `+ RGB Camera` in the SENSORS tab to add additional cameras; names are
deduplicated (`rgb_camera`, then `camera_2`, `camera_3`, …). Each enabled camera
gets its own `image_channels` entry and its own observation key
(`image` for `rgb_camera`, `image_<name>` otherwise). The **In Observations**
toggle controls whether a camera contributes an image channel without removing
it from the suite.

![SENSORS tab with a second camera configured and its observation-channel toggle](../screenshots/inspector-sensors-multi-camera.png)

### Enable / disable

Each sensor row has an `enabled` toggle. Disabled sensors stay in
`sensor_configs` (still serialized) but are skipped by `build_from_configs`,
excluded from `sensor_names`, and removed from `image_channels`. The **Duplicate
Sensor** and **Remove Sensor** actions operate on the currently selected config.

![Inspector SENSORS tab showing the four-sensor default suite and per-sensor edit rows](../screenshots/inspector-sensors.png)

## Serialized vs. runtime defaults — honest caveat

`SensorConfig` defaults (`sim_env/sensor_config.py`) differ from the runtime
class constructor defaults (`sim_core/sensors/`), because `build_sensor()`
always passes `local_pos` explicitly. If you instantiate the runtime classes
directly instead of going through `SensorConfig`, you get different mounts:

| Sensor | Param | Serialized default | Runtime ctor default |
|---|---|---|---|
| `lidar_rays` | `local_pos` | `[0, 0, 1.2]` | `[1.5, 0, 0.5]` |
| `imu` | `local_pos` | `[0, 0, 0.5]` | `[0, 0, 0.3]` |
| `imu` | `update_frequency_hz` | 100 Hz | 100 Hz (but default suite builds it at 60 Hz) |

For UI/file-authored projects the serialized column is authoritative. For
hand-built `SensorManager` suites the runtime column applies.

## See also

- [Environment Inspector — SENSORS tab](../user-guide/environment-inspector.md#sensors)
- [Vehicle dynamics model](../vehicle-dynamics/vehicle-model.md)
- [RL overview — observation pipeline](../reinforcement-learning/rl-overview.md)
- [Project schema — `agent.sensor_configs`](../configuration/project-schema.md)
