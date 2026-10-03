---
noteId: "1acfaab0bf6711f1a29f1fbaabbd87c8"
tags: []

---

# sim_core Fact Sheet (audited 2026-10-03)

Units SI (kg, m, s, N, rad). Package exports `FixedClock`, `Vec2`, `Vec3`, `OBB2D`.

## Vehicle dynamics (`sim_core/vehicle/`)

- Single-track (bicycle) dynamic model — per-axle aggregated lateral forces; lateral load transfer intentionally omitted (vehicle_model.py:174-179).
- Tire model: saturating-tanh lateral force with friction-ellipse coupling (:188-210). Longitudinal demand consumes grip first; `Fy = -lat_cap * tanh(C_alpha * alpha / max(1, cap))`.
- Longitudinal load transfer only: `transfer = m*ax*cog_height/wheelbase`, clamped ±90% static axle load (:181-186).
- Drive force linear taper to zero at `top_speed` (:157). Brake: full force opposing vx sign when |vx|>0.05, below that vx forced to 0; anti-creep zeroes vx,vy,wz if brake>0.1 and |vx|<0.1.
- `reverse_max_speed` declared but NOT enforced.
- Low-speed blend (:205-236): `kinematic_blend` active when |vx|<`low_speed_threshold` (1.5 m/s); blends toward kinematic bicycle via bounded relaxation, tau=`low_speed_relax_tau` (0.05 s). vx_eff floored at ±0.5 m/s for slip computation.
- Integration: explicit Euler substeps `h = dt/physics_substeps` (4 → 240 Hz at 60 Hz env). Controls held constant across substeps.
- `VehicleState` fields: pos (init z=0.2), yaw, pitch, roll (pitch/roll never integrated — always 0), vel_body/vel_world, yaw_rate, steering_angle, throttle, brake, accel_body (proper acceleration/specific force — matches IMU convention), accel_world, wheel_angles[FL,FR,RL,RR], wheel_speeds (visual only), is_colliding, diagnostics alpha_f/alpha_r/fy_f/fy_r/fx_f/fx_r/fz_front/fz_rear (telemetry only).

### Actions (VehicleModel.step)
| Input | Range | Meaning |
|---|---|---|
| steering_cmd | [-1,1] | negative=left, positive=right; scaled by max_steering_angle |
| throttle_cmd | [0,1] | drive force fraction |
| brake_cmd | [0,1] | brake force fraction |
| surface_friction | scalar | multiplies tire_friction |

No handbrake action exists.

### `VehicleConfig` fields (vehicle_config.py:14-58) — all user-facing, serialized in .sim.json
| Field | Type | Default | Unit | Notes |
|---|---|---|---|---|
| mass | float | 1200.0 | kg | |
| length | float | 4.2 | m | OBB half_length |
| width | float | 1.8 | m | |
| height | float | 1.4 | m | visual only |
| wheelbase | float | 2.6 | m | |
| track_width | float | 1.5 | m | wheel transforms only |
| cog_height | float | 0.45 | m | longitudinal load transfer |
| weight_dist_front | float | 0.52 | [0-1] | mass fraction over front axle |
| wheel_radius | float | 0.34 | m | |
| yaw_inertia | float | 0.0 | kg·m² | 0 = auto `Iz = mass*a*b` (a=CG→front, b=CG→rear) |
| max_steering_angle | float | 0.58 | rad (~33°) | road-wheel limit |
| steering_rate | float | 4.5 | rad/s | rate limit |
| max_drive_force | float | 6500.0 | N | at contact patches |
| max_brake_force | float | 9000.0 | N | total across axles |
| top_speed | float | 45.0 | m/s (~162 km/h) | linear power taper |
| reverse_max_speed | float | 10.0 | m/s | declared, unused in model |
| drive_force_front_fraction | float | 0.0 | [0,1] | 0=RWD default, 1=FWD, 0.5=AWD |
| brake_bias_front | float | 0.65 | fraction | |
| drag_coeff | float | 0.32 | Cd | |
| frontal_area | float | 2.1 | m² | |
| air_density | float | 1.225 | kg/m³ | |
| rolling_resistance | float | 0.015 | dimless | Frr=Crr·m·g, deadband ±0.05 m/s |
| tire_friction | float | 1.0 | µ multiplier | mu = tire_friction * surface_friction |
| cornering_stiffness_front | float | 80000.0 | N/rad | per-axle |
| cornering_stiffness_rear | float | 85000.0 | N/rad | rear>front → mild understeer |
| physics_substeps | int | 4 | — | 60Hz env → 240Hz physics |
| low_speed_threshold | float | 1.5 | m/s | kinematic blend engages below |
| low_speed_relax_tau | float | 0.05 | s | yaw-rate relaxation |

Gravity g=9.81. Drag F = 0.5·rho·Cd·A·vx·|vx|. `get_obb()` → OBB2D(half_length=length/2, half_width=width/2).

### Vehicle collision (`vehicle/collision.py`)
`CollisionResult`: collided, impact_point (=obb.center, not true contact), collider_type ∈ {"none","boundary_barrier","obstacle"}. OBB-vs-segment SAT; optional spatial-hash broadphase. Obstacle check: radius+1.0 precheck, corner-inclusion SAT both directions, segment-OBB for thin barriers. Sets `vehicle.state.is_colliding`.

## Sensors (`sim_core/sensors/`)

### BaseSensor (base_sensor.py)
Params: name, sensor_type, update_frequency_hz=30 (clamped ≥1), local_pos=Vec3(0,0,1.2), local_yaw, noise_std=0, latency_seconds=0. Update gating holds stale samples between updates; latency via history buffer (returns latest sample with t ≤ sim_time − latency); Gaussian noise elementwise. Context keys: `vehicle`, `track_queries`, `checkpoint_idx`, `obstacle_segments`.

### VehicleStateSensor (`vehicle_state`, default 60 Hz)
Output dict keys: speed (longitudinal vx), vel_x, vel_y, yaw, yaw_rate, accel_x, accel_y, steering_angle, throttle, brake, distance_from_center (+ = left), heading_error (rad), is_on_road (0/1), distance_to_checkpoint, track_progress_s (arc length), road_width. Noise applied to all except is_on_road.

### RaycastSensor (runtime `sensor_type="raycast_lidar"`; serialized type `lidar_rays`)
Defaults: num_rays=15 (≥1), fov_degrees=180, max_range=40 m, 30 Hz, local_pos runtime [1.5,0,0.5] vs serialized [0,0,1.2], local_yaw, noise_std, latency. 2D planar rays evenly spaced −fov/2…+fov/2; num_rays=1 → single forward ray. Output dict: `distances` float32[num_rays] clamped [0,max_range], `ranges_norm` = dist/max_range (1=clear,0=impact), num_rays, max_range. Miss → max_range, hit_point None. Hits `track.all_boundary_segments` + `context['obstacle_segments']`. Linear scan (no broadphase). Gaussian noise on distances, clipped, ranges_norm recomputed. Caches ray_visuals for rendering.

### IMUSensor (`imu`, class default 100 Hz; default suite uses 60 Hz)
Params: update_frequency_hz, accel_noise_std=0.05, gyro_noise_std=0.01, bias_drift_rate=0.001, local_pos serialized [0,0,0.5] (runtime ctor 0,0,0.3). Output: `linear_acceleration` float32[3]=[ax,ay,9.81] body proper accel + gravity Z; `angular_velocity`=[0,0,yaw_rate] — roll/pitch rates hardcoded 0. Random-walk bias drift `+= N(0, bias_drift_rate)` each sample, reset on reset(). Base noise_std/latency not exposed in ctor.

### CameraSensor (`camera_rgb`)
Defaults: 84×84, fov_degrees=75, 30 Hz, local_pos=[1.0,0,1.1], local_yaw=0, local_pitch=−0.05, noise_std, latency. Output uint8 (H,W,3) RGB. Two paths: injected offscreen renderer (real 3D frame if shape matches) else procedural fallback `_synthesize_perspective_road` (sky gradient top 45%, grass, gray road trapezoid center shifted by lat_offset*(w/18) and heading_err*(w/2.5), red curbs, dashed white centerline). fov_degrees/local_pitch stored but unused by procedural path. Noise = N(0, noise_std·255) clipped uint8.

### SensorManager
`update_all(sim_time, context, rng)` → {name: sample} honoring per-sensor rates. `create_default_sensor_suite()`: vehicle_state@60Hz, lidar_rays 15/180°/40m@30Hz, rgb_camera 84×84/75°@30Hz, imu@60Hz. `build_from_configs` calls `SensorConfig.build_sensor`.

## Track (`sim_core/track/`)

### RoadDefinition (road_definition.py:85-158)
Fields: name="Default Track", is_closed=True, control_points[], boundary_config, spawn_point, num_checkpoints=16, default_friction=1.0.
- `ControlPoint`: x, y (required), z=0.0 (elevation), width=12.0 m (full road width), banking=0.0 (carried to spline; NOT used in collision/dynamics), friction=1.0 (serialized; NOT consumed by vehicle step).
- `RoadBoundaryConfig`: left_type/right_type ∈ {guardrail, wall, curb, open, invisible} — only guardrail/wall produce barrier mesh; ALL types still get collision segments at road edge. wall_height=0.8, curb_width=0.5, curb_height=0.15, has_curbs=True, has_lane_markings=True (serialized, unused in mesh gen).
- `SpawnPoint`: x=0,y=0,z=0.2, yaw rad, initial_speed=0.
- `create_default_oval(radius_x=60, radius_y=35, width=12)`: 8 CPs, spawn (radius_x,0,0.1) yaw=π/2.

### TrackSpline (spline.py)
Catmull-Rom cubic through CPs, subdivisions=40/segment; closed wraps indices, open clamps. Arc-length reparameterization: cumulative chord → uniform resample sample_step=1.0 m (min 10 samples). SplinePoint: pos Vec3, tangent (central difference), normal (2D left normal (−y,x)), s, width, elevation=pos.z, banking. `get_closest_point(pos_2d)` → (s, lateral_offset [+left], tangent_angle, sample) — O(N) linear scan. `get_heading_error(yaw, tangent)` = normalize_angle(yaw−tangent). `sample_at_distance(s)` wraps modulo on closed; index-lookup.

### TrackMeshGenerator / GeneratedTrack (mesh_generator.py)
`generate(road_def, sample_step=1.0) → GeneratedTrack`. Road mesh: 2 verts per ring, normals +Z, UV v=s*0.1; float32 verts/normals/uvs. Curbs: raised curb_height at curb_width outside edge, alternating red/white every 2 m arc. Barriers: vertical quads to wall_height at outer curb edge, only for guardrail/wall. Collision boundary segments at edge pushed outward by curb_w when has_curbs; stored left/right/all_boundary_segments. Checkpoints: `max(4, num_checkpoints)` evenly spaced s = k/num·L; dict {index, s, pos, tangent, normal, width, gate_left, gate_right}. Broadphase SpatialHashGrid2D(cell_size=12.0) over all_boundary_segments. min_bounds/max_bounds from road verts.

### TrackSpatialQueries (track_queries.py)
`query_vehicle_pose(pos_2d, yaw)` → {s, lateral_offset, tangent_angle, heading_error, is_on_road(|lat|≤width/2), road_width, elevation}. `cast_ray(origin,dir,max_range=50,additional_segments)` → (distance, hit_point); brute-force AABB-prefiltered. `check_checkpoint_crossing(prev,curr,idx)` → (crossed,new_idx) via segment intersection (no direction check here — tracker does it).

## Collision (`sim_core/collision/`)
OBB2D SAT (corners FR/RR/RL/FL, segment intersection, point containment). SpatialHashGrid2D uniform grid cell_size=12.0 (configurable in ctor), segment index + entity index (entity OBB radius +0.5 m or point pos), `query_candidate_segments/entities` no false negatives, `remove_entity`, `get_memory_footprint`. Vehicle OBB ↔ boundary segments → "boundary_barrier"; ↔ entity OBBs (is_collidable) → "obstacle". No vehicle-vehicle. Raycast uses segments independently of collision checker.

## World (`sim_core/world/`)

### Entities (entity.py) — WorldEntity base (entity_id uuid4[:8], name, entity_type, semantic_label, pos Vec3, yaw, is_collidable; get_obb, get_boundary_segments=4 OBB edges, update(dt) no-op, to_dict/entity_from_dict).
| Class | entity_type | semantic | collidable | Params (defaults) |
|---|---|---|---|---|
| StaticObstacle | "obstacle" | hazard | True | obstacle_type∈{box,barrel,crate,rock}, 2.0×1.0×1.0 m; OBB min half-extent 0.1 |
| Barrier | "barrier" | infrastructure | True | barrier_type∈{concrete,guardrail,tire_wall}, 3.0×0.6×0.9 m |
| TrafficCone | "cone" | hazard | True | radius 0.3, height 0.75; square OBB half=radius |
| TrafficSign | "traffic_sign" | traffic_control | False | sign_type∈{stop,yield,speed_30,speed_50,speed_80,turn_left,turn_right,hazard_ahead}, 0.8×2.2 m |
| TrafficLight | "traffic_light" | traffic_control | False | state∈{green(12s),yellow(3s),red(10s)} cycles in update(dt); 0.6×3.2 m |
| CheckpointEntity | "checkpoint" | waypoint | False | index, gate_width=12.0, get_gate_endpoints() |
| SpawnEntity | "spawn" | spawn | False | initial_speed=0, default z=0.2 |
| Obstacle (legacy obstacle.py) | obstacle | hazard | True | x,y,radius,color=(220,50,50),is_sensor_visible=True |

ENTITY_CLASS_MAP: obstacle/static_obstacle, barrier, cone/traffic_cone, traffic_sign/sign, traffic_light, checkpoint, spawn.
Spawn logic: declarative only (SpawnPoint on RoadDefinition, SpawnEntity in world); actual reset via `VehicleModel.reset(pos, yaw, initial_speed)`.

### CheckpointTracker (world/checkpoint.py)
current_index (reset→1 when >1 cp; spawn at cp 0), laps_completed, total_checkpoints_passed, last_cross_time, lap_start_time, last_lap_time, best_lap_time=inf. `update(prev,curr,time)` → (checkpoint_passed, lap_completed): gate segment crossing + direction check (motion·tangent>0, rejects backward). Lap completes at index 0 (closed) or last index (open). `get_progress_fraction()` = (current_index−1)%num/num.

## Clock (`clock.py`)
`FixedClock(physics_hz=60.0, seed=42)`: dt=0.01667 s fixed; step_count, sim_time, time_scale=1.0, is_paused; `np.random.default_rng(seed)` shared with sensors. `advance_fixed_step()` returns dt. `compute_steps_for_real_frame(max_sub_steps=5)`: accumulator, elapsed capped 0.2 s (spiral-of-death guard), scaled by time_scale, 0 when paused.

## Caveats for docs (document honestly)
- Serialized-but-unused in sim_core: `reverse_max_speed`, `ControlPoint.friction`/`banking`, `has_lane_markings`, boundary type for collision, camera fov/pitch (procedural path).
- IMU reports only yaw rate; pitch/roll always 0. Lateral load transfer deliberately omitted.
- No handbrake; no VehicleConfig range validation (clamping at use sites).
- `get_closest_point`/`cast_ray` O(N) linear scans; only vehicle collision uses broadphase.
- Serialized lidar/IMU local_pos defaults differ from runtime ctor defaults.
