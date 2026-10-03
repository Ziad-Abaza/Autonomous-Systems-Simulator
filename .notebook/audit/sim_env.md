---
noteId: "63647260bf6711f1a29f1fbaabbd87c8"
tags: []

---

# sim_env Fact Sheet (audited 2026-10-03)

Dual-path architecture: legacy config path (ActionSpaceConfig, ObservationSchema, RewardEngine, TerminationEngine, DomainRandomizer, ScenarioConfig) and declarative "agent" path (AgentDefinition → CompiledActionDecoder, CompiledObservationPipeline, CompiledRewardEngine, CompiledTerminationEvaluator). When `agent` is provided, compiled pipelines take precedence; legacy engines instantiated but bypassed (reward_engine.last_breakdown/total mirrored for consumers).

## Environment lifecycle — `sim_env/environment.py`

`SimulationEnvironment.__init__` args (lines 46-120):
| arg | default |
|---|---|
| road_def | RoadDefinition.create_default_oval() |
| vehicle_config | VehicleConfig() |
| sensor_manager | _suite_for_agent(agent) or default suite |
| action_config | ActionSpaceConfig() |
| observation_schema | ObservationSchema() |
| reward_config | RewardConfig() |
| termination_config | TerminationConfig() |
| randomization_config | DomainRandomizationConfig() |
| scenario_config | ScenarioConfig() |
| agent | None (AgentDefinition) |
| episode_config | EpisodeConfiguration(random_seed=seed) |
| scenario_def | None (ScenarioDefinition) |
| physics_hz | 60.0 |
| seed | 42 |

Creates FixedClock, TrackMeshGenerator.generate(road_def), TrackSpatialQueries, VehicleModel, CheckpointTracker; if agent → compiles pipelines. `active_surface_friction = road_def.default_friction * fric_mult`. Sensors priority: explicit sensor_manager > agent.sensor_configs > default suite.

Other methods: set_agent, set_scenario (clears _scenario_spawned entities, does NOT reset), set_road_definition (regenerates mesh/queries/checkpoints), add_obstacle, add_entity, remove_entity, clear_obstacles, clear_entities.

`reset(seed=None, options=None)` → (obs, info) (233-365):
- reset_seed = seed or episode_config.random_seed; clock.reset(seed)
- DR sampling: scenario_def.randomization.sample_all(clock.rng) when enabled (keys vehicle_mass_mult, tire_friction_mult, surface_friction_mult, spawn_lateral_jitter_m, spawn_heading_jitter_deg, sensor_noise_mult) else legacy domain_randomizer
- active_surface_friction = default_friction * scen_surf_mult * surf_f
- sensor noise = scenario_def.sensor_noise_mult * noise_f (multiplies noise_std/accel/gyro noise vs snapshotted bases)
- vehicle.config.mass *= mass_f; tire_friction *= tire_f; _recompute_inertials
- Spawn = road_def.spawn_point + lateral jitter + yaw jitter; scenario_def.spawn_override keys {pos:[x,y], yaw_deg, initial_speed}; SpawnMode.CUSTOM_POSE uses episode_config.custom_spawn_pos/custom_spawn_yaw_deg/initial_speed. SpawnMode.RANDOM_CHECKPOINT declared but NEVER handled.
- scenario_def.target_speed_override → compiled speed component params.target_speed_ms + legacy reward target_speed
- scenario_def.obstacle_overrides → create_entity each reset, tagged _scenario_spawned; keys {entity_type(default cone), pos:[x,y,z], yaw, name}

`step(action)` → (obs, reward, terminated, truncated, info) (367-554):
- step() after is_done → returns (obs, 0.0, True, True, info) with info['error'], termination_reason="invalid_call_after_done"
- action decoded via compiled decoder or action_config.decode_action; vehicle.step(steer,throttle,brake, active_surface_friction)
- collisions: boundary + obstacle → is_colliding
- track_queries.query_vehicle_pose → is_on_road, lateral_offset, heading_error, s, road_width
- checkpoint update → cp_passed, lap_completed
- reward via compiled or legacy engine
- termination via compiled evaluator (reason dict) or legacy engine (reason string)
- env-level duration truncation: effective_max_duration = scenario_def.time_limit_override or episode_config.max_duration_seconds → "scenario_time_limit_exceeded" / "max_duration_exceeded"
- terminated or truncated → is_done=True; exposes last_step_reward/last_step_info

info dict keys (_build_info_dict 645-680): step, sim_time, terminated, truncated, termination_reason (str), termination_info (dict), reward_breakdown (dict), total_reward, speed, lateral_offset, heading_error, road_width, is_on_road, is_colliding, action_valid, action_error, checkpoints_passed, current_checkpoint, laps_completed, lap_progress, last_action [steer,throttle,brake].

Semantics: terminated = task end; truncated = horizon limit; both set is_done.

## Observation

### Declarative ObservationSpaceDefinition (observation_designer.py:70-315)
ObservationChannelConfig: name, channel_type "scalar"|"vector"|"image", shape=[1], dtype "float32"|"uint8", range_low=[-1.0], range_high=[1.0], normalization ∈ {none, scale, min_max, standardized(declared, NOT implemented), clip}, norm_params{}, source_sensor="vehicle_state", source_key="speed", category ∈ {agent_observation, debug_telemetry, oracle_ground_truth}, enabled=True, description.
ObservationSpaceDefinition: channels[], flatten_vector=True, include_image_channel=False (legacy), image_channels[{name,shape}] (authoritative multi-camera).

create_default_space() = 23-dim vector: speed ÷45; velocity_body[2] (scale_x 45, scale_y 10); yaw_rate ÷3; steering_angle ÷0.6; distance_from_center ÷half-width; heading_error ÷π; distance_to_checkpoint ÷100; lidar_ranges[15] raw ranges_norm.
image_channel_specs(): image_channels else [{rgb_camera,[84,84,3]}] if legacy flag.
validate_no_leakage(): ERROR if enabled channel is debug_telemetry or oracle_ground_truth; compiled pipeline raises ValueError.
export_schema() → {flatten_vector, vector_dimension, include_image, image_channels, num_channels, channels[{name,type,shape,dtype,range{low,high},normalization,norm_params,source:"sensor:key",category,description}]}.

CompiledObservationPipeline: preallocated float32 buffer; flattened → ndarray; with images → {"vector", "image"} or {"vector", "image_<sensor_name>"} ("image" only for sensor named exactly rgb_camera); dict mode keys=channel names. NaN→0, ±Inf→±1. Missing lidar → ones(15). Missing image → zeros uint8.

### Legacy ObservationSchema (spaces.py:135-182)
include_speed/velocity/yaw_rate/steering_angle/distance_from_center/heading_error/distance_to_checkpoint/lidar_rays=True, include_camera_rgb=False, flatten_vector=True. compute_vector_dim(15)=23 default. Same normalization constants. Dict-mode keys: speed, vel_body, yaw_rate, steering_angle, distance_from_center, heading_error, distance_to_checkpoint, lidar_ranges, camera_rgb.

### observation_contract.py
FieldClass=ChannelCategory; DIAGNOSTIC="diagnostic". diagnostic_state_contract() GET_STATE fields: speed, pos, yaw, sim_time, total_reward, reward_breakdown. channel_classification_map, agent_observation_fields.

## Action

### Declarative ActionSpaceDefinition (action_designer.py:71-182)
ActionType continuous|discrete. ActionChannelConfig: name, channel_type, min_val=-1, max_val=1, default_val=0, scaling=1.0, dead_zone=0 (|dev-from-default|<dz → default), rate_limit=0 (units/s, 0=unlimited), description. DiscreteActionOption{name, values[]}.

create_default_vehicle_action_space() — 3 channels:
- steering [-1,1], default 0, dead_zone 0.02, rate_limit 4.0/s
- throttle [0,1], dead_zone 0.01, rate_limit 5.0/s
- brake [0,1], dead_zone 0.01, rate_limit 6.0/s
No handbrake. Discrete options: Coast [0,0,0], Accelerate [0,0.7,0], Brake [0,0,0.8], Steer Left+Throttle [-0.6,0.4,0], Steer Right+Throttle [0.6,0.4,0].
export_schema() → {space_type, num_channels, channels[{name,type,min,max,default,scaling,dead_zone,rate_limit,description}], num_discrete_actions, discrete_actions[{name,values}]}.

CompiledActionDecoder: sanitize NaN→0 → target*scalings → deadzone → rate limit (rl*dt, continuous only, uses prev_action, reset per episode) → clip [mins,maxs]. validate_action errors on None/empty/NaN/Inf/undersized.

### Legacy ActionSpaceConfig (spaces.py:17-132)
type CONTINUOUS|DISCRETE, continuous_low=[-1,0,0], high=[1,1,1] (order steer,throttle,brake), discrete_actions same 5×3. decode: NaN→0, Inf→±1, clip; NO deadzone/rate limiting.

## Reward

### Legacy RewardConfig (reward_engine.py:13-68) — validated ≥0
weight_progress=1.0, weight_centering=0.5, weight_speed=0.2, target_speed=20.0 m/s, weight_heading=0.3, weight_action_smoothness=0.05, checkpoint_bonus=10.0, lap_completion_bonus=100.0, collision_penalty=50.0, off_road_penalty=25.0, backward_penalty=1.0.
Formulas (101-193); delta_s wrap-corrected, discarded if |Δs|>5 m (teleport guard):
- progress = delta_s·weight_progress
- backward = −backward_penalty if delta_s<−0.05 or |heading_error|>0.6π
- centering = (1−min(1,|lat|/half_w))·weight_centering
- speed = min(1,max(0,speed/target_speed))·weight_speed
- heading = cos(heading_error)·weight_heading
- smoothness = −(Δsteer)²·weight_action_smoothness
- checkpoint/lap = bonus on event; collision/off_road = −penalty
- breakdown keys: progress, centering, speed, heading, smoothness, checkpoint, lap, collision, off_road, backward, total

### Declarative RewardFunctionDefinition (reward_designer.py)
components: [{component_id, name, component_type, weight=1.0, enabled, params{}, description}]. FalloffType linear|quadratic|exponential. Aggregation=weighted_sum.

create_default_racing_reward() — 11 components:
| component_id | type | weight | params |
|---|---|---|---|
| progress | progress | 1.0 | max_step_delta_m 5.0 |
| centering | centerline | 0.5 | max_distance_m 6.0, falloff linear |
| speed | speed | 0.2 | target_speed_ms 20.0, tolerance 5.0 |
| heading | heading | 0.3 | — |
| smooth_steer | smooth_steer | −0.05 | — |
| checkpoint | checkpoint | 10.0 | — |
| completion | completion | 100.0 | — |
| collision | collision | −50.0 | — |
| off_road | off_road | −25.0 | — |
| reverse | reverse | −1.0 | heading_threshold_deg 100.0 |
| time_penalty | time_penalty | −0.01 | — |

Raw values: progress→Δs; centerline→1−norm / 1−norm² / exp(−3·norm); speed→min(1,speed/target); heading→cos(err); smooth_steer→−(Δsteer)²; checkpoint/completion/collision/off_road→0/1 events; reverse→1 if Δs<−0.05 or |err|>threshold; time_penalty→1.0. Contribution=raw×weight. last_breakdown={component_id: contrib, total}; last_detailed_breakdown={id:{raw,weight,contrib}}.
NOTE: params max_step_delta_m & tolerance declared but not consumed (5.0 guard hardcoded).

## Termination

### Legacy TerminationConfig (termination_engine.py:11-34)
terminate_on_collision=True, terminate_on_off_road=True, terminate_on_wrong_direction=True, wrong_direction_max_angle_deg=120.0, terminate_on_lap_completion=False, laps_to_complete=1, max_episode_steps=5000, max_seconds_without_checkpoint=30.0.
Reason strings: "collision","off_road","wrong_direction" (term); "course_completed" (open track ≥1 lap, term); "lap_completed" (if enabled, term); "checkpoint_timeout" (trunc); "max_steps_exceeded" (trunc); "running".

### Declarative (termination_designer.py)
TerminationConditionType: collision, off_road, wrong_direction, course_completion, max_steps, simulation_timeout, checkpoint_timeout, custom_threshold (declared, NOT evaluated).
TerminationRuleConfig: rule_id, name, condition_type, enabled, is_truncation, params, description.
create_default_racing_termination() — 6 rules: term_collision(collision,on), term_off_road(on), term_wrong_direction(on, max_angle_deg 120), term_completion(off, target_laps 1), trunc_max_steps(on, max_steps 5000, trunc), trunc_stuck(on, checkpoint_timeout, max_seconds 30, trunc).
CompiledTerminationEvaluator tracks episode_steps/time/time_since_checkpoint. Reason dict {rule_id, condition, reason, is_truncation, step, sim_time}; reason = rule_id minus term_/trunc_ prefix → "collision","off_road","wrong_direction","completion","max_steps","stuck". Idle → "running". Env adds "max_duration_exceeded"/"scenario_time_limit_exceeded".

## Scenarios

### Legacy ScenarioConfig (scenarios.py:12-44)
name="Standard Day", time_of_day day|dusk|night, weather clear|rain|fog, ambient_light=1.0 (0.1-1.0), surface_friction_mult=1.0, obstacles[], difficulty_level=1. Presets: day_clear, wet_rain (dusk/rain, 0.6 light, 0.7 friction), night_fog (night/fog, 0.25 light, 0.9 friction).

### ScenarioDefinition (scenario_designer.py:14-70)
scenario_id, name, description, weather, time_of_day, ambient_light, surface_friction_mult, target_speed_override, time_limit_override, sensor_noise_mult, spawn_override{pos,yaw_deg,initial_speed}, obstacle_overrides[{entity_type,pos,yaw,name}], randomization (DomainRandomizationDefinition).
get_standard_scenarios() — 6 built-ins: basic_lane_following (speed 18), high_speed_racing (35), wet_adverse_weather (rain/dusk, 0.6, 0.65, 16), obstacle_evasion (cone@[30,10], barrier@[-25,-15,0.4]), sensor_noise_challenge (fog/night, 0.3, noise 2.5, friction 0.9), full_domain_randomization (rand enabled, seed 1337).

## Domain randomization

### Legacy DomainRandomizationConfig (domain_randomizer.py)
enabled=False; mass_range(0.85,1.15), tire_friction_range(0.80,1.20), surface_friction_range(0.75,1.10), sensor_noise_multiplier_range(0.5,2.0), spawn_lateral_jitter_m=1.0 (±uniform), spawn_heading_jitter_deg=10.0 (±uniform → radians). sample_parameters → mass_factor, tire_friction_factor, surface_friction_factor, sensor_noise_factor, spawn_lateral_jitter, spawn_heading_jitter (1.0/0.0 when disabled).

### Declarative DomainRandomizationDefinition (randomization_designer.py)
enabled=False, global_seed=42, parameters{param_name: RandomParamConfig{param_name, distribution∈{fixed,uniform,normal}, param1 (fixed/min/mean), param2 (max/std), clip_min, clip_max}}.
create_default(): vehicle_mass_mult U[0.85,1.15] clip[0.5,2.0]; tire_friction_mult U[0.80,1.20] clip[0.4,2.0]; surface_friction_mult U[0.75,1.10] clip[0.3,2.0]; sensor_noise_mult U[0.5,2.0] clip[0,5.0]; spawn_lateral_jitter_m U[−1,1] clip[−3,3]; spawn_heading_jitter_deg N(0,5) clip[−30,30]. Uses clock.rng else default_rng(global_seed).

## Validator (validator.py)
IssueSeverity ERROR|WARNING|INFO. ValidationIssue{severity,subsystem,message,remediation}; ValidationReport.is_valid_for_rl (False if any ERROR), to_dict → {is_valid_for_rl, error_count, warning_count, info_count, issues}.
EnvironmentValidator.validate(road_def, agent, entities, available_sensors, track_mesh) subsystems: Agent, Track, Spawn, Action, Observation, Reward, Termination, Sensors, Environment. Checks include: no agent→ERROR; empty agent_id→ERROR; <3 CPs→ERROR; CP pairs<0.5m→ERROR; num_checkpoints<2→ERROR; spawn off road→ERROR; spawn heading >45°→WARNING; spawn within 3.5m of collidable→ERROR; discrete<2 options→ERROR; 0 channels→ERROR; min>=max→ERROR; NaN bounds→ERROR; default outside bounds→WARNING; obs channel source_sensor not attached→ERROR; image camera not enabled→ERROR; leakage→ERROR "SECURITY LEAKAGE"; no progress reward→WARNING; NaN weight→ERROR; no termination rules→ERROR; no truncation rule→WARNING; duplicate sensor names→ERROR; INFO gate count, obs dim, "deterministic fixed 60 Hz stepping", open route INFO, track<20m WARNING.

## Templates (templates.py)
list_templates() ids: empty, basic_driving (default fallback), straight_sprint, hairpin, slalom, lane_following, obstacle_avoidance. Each → EnvironmentProject.
- empty: open 100m straight, 4 checkpoints
- basic_driving: default oval 65×40 w12 "Proving Ground Oval"
- straight_sprint: open 300m, 6 checkpoints
- hairpin: closed 10 CPs, 16 checkpoints
- slalom: open 200m, 16m wide, 8 checkpoints, 5 cones at x=40+32i ±4.0
- lane_following: closed serpentine 10 CPs, 20 checkpoints, centering.weight=1.0, heading.weight=0.8
- obstacle_avoidance: oval + 4 entities (3 cones, 1 barrier), collision.weight=−100.0

## Versioning (versioning.py, serializer.py)
EnvironmentVersion 1.0.0, M.m.p. compute_fingerprint = SHA-256 of JSON sort_keys minus keys {version,timestamp,last_saved,author,environment_version,schema_version,fingerprint}. compare_configurations → ConfigurationDiffReport over 6 subsystems (Track, Observation, Action, Reward, Termination, Entities & Scenario).
EnvironmentProject SCHEMA_VERSION="2.0.0", V3="3.0.0". Top keys: schema_version, environment_version, name, road_definition, vehicle_config, action_config, observation_schema, reward_config, termination_config, randomization_config, scenario_config, entities, agent, episode_config, scenario_def, [curriculum], [experiment_config], fingerprint.
Migration: defaults schema_version "1.0.0"; entities ← entities|obstacles|scenario_config.obstacles; missing agent→default; missing episode_config synthesized; missing scenario_def→basic_lane_following; coerced "2.0.0" unless already 2/3. save(auto_increment) bumps env_version patch on fingerprint change.

## Curriculum (curriculum.py)
CurriculumStage: stage_id, name, description, scenario_id (standard scenarios), target_metric (mean_return default; options mean_return, lap_completion_rate, collision_rate), advancement_threshold=100.0, min_episodes=50, environment_overrides{target_speed→…}.
create_default() 5 stages: S1 Lane Keeping (basic_lane_following, mean_return≥50, 20ep, target_speed 12), S2 High Speed (high_speed_racing, ≥120, 30ep, speed 22), S3 Obstacles (obstacle_evasion, lap_completion_rate≥0.85, 40ep), S4 Adverse (wet_adverse_weather, ≥0.80, 40ep), S5 Full DR (≥0.90, 50ep). Serialized under "curriculum" key. Runtime advancement in sim_experiment/curriculum_runtime.py, NOT in SimulationEnvironment.

## Episode config & agent

### EpisodeConfiguration (episode_config.py)
max_duration_seconds=60.0, max_steps=3600, spawn_mode=TRACK_SPAWN, initial_speed=0.0, custom_spawn_pos=(0,0,0.2), custom_spawn_yaw_deg=0, spawn_lateral_jitter_m=0, spawn_heading_jitter_deg=0, random_seed=42, reset_behavior=HARD_RESET, auto_reset_on_done=False.
SpawnMode: track_spawn|random_cp|custom_pose — only custom_pose implemented. ResetBehavior hard|soft — config only. auto_reset_on_done & max_steps serialized but NOT consumed by step().

### AgentSpawnConfig (agent.py)
spawn_mode="default" (default|custom|checkpoint — comment only, not wired), pos, yaw_deg, initial_speed, lateral_jitter_m, heading_jitter_deg.

### AgentDefinition (agent.py:51-133)
agent_id="agent_01", name, entity_type="vehicle", entity_id="vehicle_01", sensor_configs[] (source of truth), sensor_names (derived, back-compat), observation_space, action_space, reward_function, termination_rules, spawn_config. Serialized under "agent".

### SensorConfig (sensor_config.py)
Types: camera_rgb, lidar_rays, imu, vehicle_state. Fields name, sensor_type, enabled=True, params{}.
Defaults: camera_rgb{w84,h84,fov75,30Hz,local_pos[1,0,1.1],yaw0,pitch−0.05,noise0,latency0}; lidar_rays{15 rays,180°,40m,30Hz,pos[0,0,1.2]}; imu{100Hz,accel_noise0.05,gyro0.01,bias_drift0.001,pos[0,0,0.5]}; vehicle_state{60Hz,noise0,latency0}. default_suite_configs() = 4 sensors (camera name "rgb_camera").

## experiment.py, export.py
ExperimentConfig: experiment_id, name, environment_name, scenario_id, seed=42, algorithm PPO|SAC|DQN|Custom, total_timesteps=50000, rollout_steps=1024, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, batch_size=256, num_epochs=4, eval_frequency_steps=5000, eval_episodes=5, checkpoint_frequency_steps=10000, output_directory, hyperparameters{}.
TrainingExporter.export_training_bundle → environment.json, scenario.json, experiment.json, run_gym_training.py (generated PPO runner using EnvironmentProject.from_dict + PPORunner).

## Dead/unused config surface (document honestly)
SpawnMode.RANDOM_CHECKPOINT, ResetBehavior, auto_reset_on_done, AgentSpawnConfig fields, NormalizationType.STANDARDIZED, TerminationConditionType.CUSTOM_THRESHOLD, episode_config.max_steps, reward params max_step_delta_m & tolerance.
