"""
High-performance Gymnasium-compliant Simulation Environment.
Integrates world physics, modular vehicle dynamics, sensors, reward engine,
termination logic, scenario conditions, and Phase 3 Declarative Agent pipeline.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np

from sim_core.clock import FixedClock
from sim_core.math_utils import Vec2, Vec3
from sim_core.track.road_definition import RoadDefinition
from sim_core.track.mesh_generator import TrackMeshGenerator, GeneratedTrack
from sim_core.track.track_queries import TrackSpatialQueries
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.vehicle.vehicle_model import VehicleModel
from sim_core.vehicle.collision import VehicleCollisionChecker
from sim_core.sensors.sensor_manager import SensorManager
from sim_core.sensors.camera_sensor import CameraSensor
from sim_core.sensors.raycast_sensor import RaycastSensor
from sim_core.world.checkpoint import CheckpointTracker
from sim_core.world.obstacle import Obstacle
from sim_core.world.entity import WorldEntity

from sim_env.spaces import ActionSpaceConfig, ObservationSchema, ActionSpaceType
from sim_env.reward_engine import RewardEngine, RewardConfig
from sim_env.termination_engine import TerminationEngine, TerminationConfig
from sim_env.domain_randomizer import DomainRandomizer, DomainRandomizationConfig
from sim_env.scenarios import ScenarioConfig
from sim_env.agent import AgentDefinition
from sim_env.episode_config import EpisodeConfiguration, SpawnMode, ResetBehavior
from sim_env.scenario_designer import ScenarioDefinition
from sim_env.action_designer import ActionSpaceDefinition, CompiledActionDecoder
from sim_env.observation_designer import ObservationSpaceDefinition, CompiledObservationPipeline
from sim_env.reward_designer import RewardFunctionDefinition, CompiledRewardEngine
from sim_env.termination_designer import TerminationDefinition, CompiledTerminationEvaluator


class SimulationEnvironment:
    """
    Core AI Environment integrating Simulation + Agent Sensors + Rewards + Termination.
    Follows standard Gymnasium reset/step lifecycle with compiled zero-overhead pipelines.
    """
    def __init__(
        self,
        road_def: Optional[RoadDefinition] = None,
        vehicle_config: Optional[VehicleConfig] = None,
        sensor_manager: Optional[SensorManager] = None,
        action_config: Optional[ActionSpaceConfig] = None,
        observation_schema: Optional[ObservationSchema] = None,
        reward_config: Optional[RewardConfig] = None,
        termination_config: Optional[TerminationConfig] = None,
        randomization_config: Optional[DomainRandomizationConfig] = None,
        scenario_config: Optional[ScenarioConfig] = None,
        agent: Optional[AgentDefinition] = None,
        episode_config: Optional[EpisodeConfiguration] = None,
        scenario_def: Optional[ScenarioDefinition] = None,
        physics_hz: float = 60.0,
        seed: int = 42
    ):
        # Clock & Determinism
        self.clock = FixedClock(physics_hz=physics_hz, seed=seed)

        # Track and Road Geometry
        self.road_def = road_def or RoadDefinition.create_default_oval()
        self.track = TrackMeshGenerator.generate(self.road_def)
        self.track_queries = TrackSpatialQueries(self.track)

        # Vehicle
        self.base_vehicle_config = vehicle_config or VehicleConfig()
        self.vehicle = VehicleModel(self.base_vehicle_config)

        # World Entities & Obstacles
        self.entities: List[WorldEntity] = []
        self.obstacles: List[Any] = []
        self.checkpoint_tracker = CheckpointTracker(self.track)

        # Sensors — explicit manager wins; then the agent's declarative
        # sensor suite; then the legacy hardcoded default.
        self._explicit_sensor_manager = sensor_manager is not None
        self.sensors = sensor_manager or self._suite_for_agent(agent)

        # Phase 3 Declarative Subsystems
        self.agent: Optional[AgentDefinition] = agent
        self.episode_config: EpisodeConfiguration = episode_config or EpisodeConfiguration(random_seed=seed)
        self.scenario_def: Optional[ScenarioDefinition] = scenario_def

        # Legacy RL Engines & Schemas (maintained for full backward compatibility)
        self.action_config = action_config or ActionSpaceConfig()
        self.observation_schema = observation_schema or ObservationSchema()
        self.reward_engine = RewardEngine(reward_config or RewardConfig())
        self.termination_engine = TerminationEngine(termination_config or TerminationConfig())
        self.domain_randomizer = DomainRandomizer(randomization_config or DomainRandomizationConfig())
        self.scenario = scenario_config or ScenarioConfig()

        # Compiled Phase 3 pipelines
        self.compiled_action_decoder: Optional[CompiledActionDecoder] = None
        self.compiled_obs_pipeline: Optional[CompiledObservationPipeline] = None
        self.compiled_reward_engine: Optional[CompiledRewardEngine] = None
        self.compiled_termination_evaluator: Optional[CompiledTerminationEvaluator] = None

        if self.agent is not None:
            self._compile_agent_pipelines()

        # Snapshot base sensor noise levels so scenario multipliers are applied
        # idempotently on every reset instead of compounding.
        self._snapshot_sensor_noise()

        # Active physical properties
        fric_mult = self.scenario_def.surface_friction_mult if self.scenario_def else self.scenario.surface_friction_mult
        self.active_surface_friction = self.road_def.default_friction * fric_mult

        # History tracking
        self.current_step = 0
        self.is_done = False
        self.last_action = [0.0, 0.0, 0.0]
        self.last_termination_reason_dict: Dict[str, Any] = {"reason": "running"}
        self._scenario_time_limit_s: Optional[float] = None

    @staticmethod
    def _suite_for_agent(agent: Optional[AgentDefinition]) -> "SensorManager":
        """Builds the runtime sensor suite from an agent's declarative
        sensor configs (falls back to the default suite when absent)."""
        cfgs = getattr(agent, "sensor_configs", None)
        if agent is not None and cfgs:
            return SensorManager.build_from_configs(cfgs)
        return SensorManager.create_default_sensor_suite()

    def _snapshot_sensor_noise(self) -> None:
        self._base_sensor_noise: Dict[str, Dict[str, float]] = {}
        for s_name, sensor in self.sensors.sensors.items():
            base: Dict[str, float] = {}
            for attr in ("noise_std", "accel_noise_std", "gyro_noise_std"):
                if hasattr(sensor, attr):
                    base[attr] = float(getattr(sensor, attr))
            if base:
                self._base_sensor_noise[s_name] = base

    def _apply_sensor_noise_multiplier(self, mult: float) -> None:
        """Scales all sensor noise levels relative to their captured base values."""
        for s_name, base in self._base_sensor_noise.items():
            sensor = self.sensors.sensors.get(s_name)
            if sensor is None:
                continue
            for attr, base_val in base.items():
                setattr(sensor, attr, base_val * mult)

    def _compile_agent_pipelines(self) -> None:
        """Compiles authoring agent spaces and functions into zero-overhead runtime engines."""
        if self.agent is not None:
            self._assert_obs_channel_sensor_coverage()
            self.compiled_action_decoder = self.agent.action_space.compile_decoder()
            self.compiled_obs_pipeline = self.agent.observation_space.compile_pipeline()
            self.compiled_reward_engine = self.agent.reward_function.compile_engine()
            self.compiled_termination_evaluator = self.agent.termination_rules.compile_evaluator()

    def _assert_obs_channel_sensor_coverage(self) -> None:
        """Every enabled observation channel must map to an attached sensor.

        Runtime must agree with the validator: a channel whose source
        sensor is disabled or absent would otherwise silently produce a
        constant/zeroed column — an invalid observation contract."""
        attached = set(self.sensors.sensors.keys())
        missing = []
        for ch in self.agent.observation_space.channels:
            if ch.enabled and ch.source_sensor not in attached:
                missing.append(f"channel '{ch.name}' -> sensor '{ch.source_sensor}'")
        for spec in self.agent.observation_space.image_channel_specs():
            if spec.get("name") not in attached:
                missing.append(f"image channel -> sensor '{spec.get('name')}'")
        if missing:
            raise ValueError(
                "Observation channels reference unattached/disabled sensors: "
                + "; ".join(missing)
                + ". Disable the channel or attach/enable the sensor.")

    def set_agent(self, agent: AgentDefinition) -> None:
        """Sets active agent definition, rebuilds its declarative sensor
        suite, and compiles runtime pipelines."""
        self.agent = agent
        if not self._explicit_sensor_manager:
            self.sensors = self._suite_for_agent(agent)
            self._snapshot_sensor_noise()
        self._compile_agent_pipelines()

    def set_scenario(self, scenario_def: Optional[ScenarioDefinition]) -> None:
        """
        Replaces the active ScenarioDefinition for subsequent resets.
        Removes entities previously spawned from scenario obstacle_overrides;
        does NOT reset the environment — callers decide when the new scenario
        takes effect (typically immediately via reset()).
        """
        self._clear_scenario_entities()
        self.scenario_def = scenario_def

    def _clear_scenario_entities(self) -> None:
        """Removes scenario-spawned entities from entities/obstacles/broadphase."""
        spawned = [e for e in self.entities
                   if getattr(e, '_scenario_spawned', False)]
        for ent in spawned:
            self.entities.remove(ent)
            if ent in self.obstacles:
                self.obstacles.remove(ent)
            bp = getattr(self.track, 'broadphase', None)
            if bp is not None:
                bp.remove_entity(ent)

    def set_road_definition(self, road_def: RoadDefinition) -> None:
        """Updates road definition and regenerates 3D mesh and collision bounds."""
        self.road_def = road_def
        self.track = TrackMeshGenerator.generate(self.road_def)
        self.track_queries = TrackSpatialQueries(self.track)
        self.checkpoint_tracker = CheckpointTracker(self.track)
        if self.track.broadphase:
            for ent in self.entities:
                self.track.broadphase.insert_entity(ent)

    def add_obstacle(self, obstacle: Any) -> None:
        if obstacle not in self.obstacles:
            self.obstacles.append(obstacle)
        if obstacle not in self.entities:
            self.entities.append(obstacle)
        if self.track.broadphase:
            self.track.broadphase.insert_entity(obstacle)

    def add_entity(self, entity: WorldEntity) -> None:
        if entity not in self.entities:
            self.entities.append(entity)
        if getattr(entity, 'is_collidable', False) and entity not in self.obstacles:
            self.obstacles.append(entity)
        if self.track.broadphase:
            self.track.broadphase.insert_entity(entity)

    def remove_entity(self, entity_id: str) -> bool:
        to_remove = [e for e in self.entities if getattr(e, 'entity_id', '') == entity_id]
        if to_remove:
            for ent in to_remove:
                self.entities.remove(ent)
                if ent in self.obstacles:
                    self.obstacles.remove(ent)
            return True
        return False

    def clear_obstacles(self) -> None:
        self.obstacles.clear()
        self.entities = [e for e in self.entities if getattr(e, 'entity_type', '') != 'obstacle']

    def clear_entities(self) -> None:
        self.entities.clear()
        self.obstacles.clear()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, Dict[str, Any]], Dict[str, Any]]:
        """
        Resets environment to initial state, returning (initial_observation, info).
        """
        reset_seed = seed if seed is not None else self.episode_config.random_seed
        self.clock.reset(seed=reset_seed)

        self.current_step = 0
        self.is_done = False
        self.last_action = [0.0, 0.0, 0.0]
        self.last_termination_reason_dict = {"reason": "reset", "step": 0, "sim_time": 0.0}

        # Domain Randomization (Scenario Def or Legacy Config)
        # The sampling rng is seeded per reset from BOTH the episode seed
        # and the authored randomization.global_seed — per-reset variance
        # without bypassing the authored seed.
        if self.scenario_def and self.scenario_def.randomization.enabled:
            dr_rng = np.random.default_rng(np.random.SeedSequence(
                [int(reset_seed),
                 int(self.scenario_def.randomization.global_seed)]))
            dr_params = self.scenario_def.randomization.sample_all(dr_rng)
            mass_f = dr_params.get("vehicle_mass_mult", 1.0)
            tire_f = dr_params.get("tire_friction_mult", 1.0)
            surf_f = dr_params.get("surface_friction_mult", 1.0)
            lat_jit = dr_params.get("spawn_lateral_jitter_m", 0.0)
            yaw_jit = math.radians(dr_params.get("spawn_heading_jitter_deg", 0.0))
            noise_f = dr_params.get("sensor_noise_mult", 1.0)
        else:
            legacy_dr = self.domain_randomizer.sample_parameters(self.clock.rng)
            mass_f = legacy_dr['mass_factor']
            tire_f = legacy_dr['tire_friction_factor']
            surf_f = legacy_dr['surface_friction_factor']
            lat_jit = legacy_dr['spawn_lateral_jitter']
            yaw_jit = legacy_dr['spawn_heading_jitter']
            noise_f = legacy_dr.get('sensor_noise_factor', 1.0)

        scen_surf_mult = self.scenario_def.surface_friction_mult if self.scenario_def else self.scenario.surface_friction_mult
        self.active_surface_friction = self.road_def.default_friction * scen_surf_mult * surf_f

        # Scenario sensor noise override: scale each sensor's base noise levels.
        scen_noise_mult = self.scenario_def.sensor_noise_mult if self.scenario_def else 1.0
        self._apply_sensor_noise_multiplier(scen_noise_mult * noise_f)

        # Scenario episode time limit override (consumed by step() truncation check).
        self._scenario_time_limit_s: Optional[float] = (
            self.scenario_def.time_limit_override if self.scenario_def else None
        )

        # Vehicle parameters
        self.vehicle.config.mass = self.base_vehicle_config.mass * mass_f
        self.vehicle.config.tire_friction = self.base_vehicle_config.tire_friction * tire_f
        self.vehicle._recompute_inertials()

        # Remove entities spawned by scenario obstacle_overrides on previous
        # resets — they are re-created below from the CURRENT scenario_def,
        # so obstacles never accumulate across resets or scenario switches.
        self._clear_scenario_entities()

        # Compute initial spawn pose
        sp = self.road_def.spawn_point
        spawn_yaw = sp.yaw + yaw_jit
        cos_y = math.cos(spawn_yaw)
        sin_y = math.sin(spawn_yaw)
        norm_x = -sin_y
        norm_y = cos_y
        spawn_x = sp.x + norm_x * lat_jit
        spawn_y = sp.y + norm_y * lat_jit
        init_speed = sp.initial_speed

        if self.scenario_def and self.scenario_def.spawn_override:
            so = self.scenario_def.spawn_override
            if "pos" in so:
                spawn_x, spawn_y = so["pos"][0], so["pos"][1]
            if "yaw_deg" in so:
                spawn_yaw = math.radians(so["yaw_deg"])
            if "initial_speed" in so:
                init_speed = float(so["initial_speed"])

        if self.episode_config.spawn_mode == SpawnMode.CUSTOM_POSE:
            c_pos = self.episode_config.custom_spawn_pos
            spawn_x, spawn_y = c_pos[0], c_pos[1]
            spawn_yaw = math.radians(self.episode_config.custom_spawn_yaw_deg)
            init_speed = self.episode_config.initial_speed

        # The vehicle model is planar: pos.z is a ground-reference height
        # consumed by rendering, cameras, and vehicle-mounted sensors.
        # Anchor it to the road surface elevation under the spawn pose so
        # the car never spawns sunk into or floating above elevated road
        # (authored sp.z remains the fallback when no spline exists).
        spawn_z = sp.z
        if self.track.spline.samples:
            _, _, _, spawn_sample = self.track.spline.get_closest_point(
                Vec2(spawn_x, spawn_y))
            spawn_z = spawn_sample.pos.z

        self.vehicle.reset(
            pos=Vec3(spawn_x, spawn_y, spawn_z),
            yaw=spawn_yaw,
            initial_speed=init_speed
        )

        # Mirror authored scenario visual fields onto the legacy scenario
        # so the renderer/HUD path (which reads self.scenario) sees the
        # effective values — explicit ambient_light wins; otherwise it is
        # derived from time_of_day.
        if self.scenario_def:
            self.scenario.weather = self.scenario_def.weather
            self.scenario.time_of_day = self.scenario_def.time_of_day
            self.scenario.ambient_light = (
                self.scenario_def.ambient_light
                if self.scenario_def.ambient_light != 1.0
                else {"day": 1.0, "dusk": 0.6, "night": 0.25}.get(
                    self.scenario_def.time_of_day, 1.0))

        # Apply scenario target speed override to reward engines. The
        # baseline is restored FIRST every reset so the override never
        # leaks into a later scenario that does not set one.
        if not hasattr(self, "_base_speed_targets"):
            self._base_speed_targets = []
            if self.compiled_reward_engine:
                for comp in self.compiled_reward_engine.active_components:
                    if comp.component_type == "speed":
                        self._base_speed_targets.append(
                            (comp, comp.params.get("target_speed_ms")))
        for comp, base in self._base_speed_targets:
            if base is not None:
                comp.params["target_speed_ms"] = base
        if hasattr(self.reward_engine, 'config'):
            if not hasattr(self, "_base_legacy_target_speed"):
                self._base_legacy_target_speed = getattr(
                    self.reward_engine.config, "target_speed", None)
            if self._base_legacy_target_speed is not None:
                self.reward_engine.config.target_speed = \
                    self._base_legacy_target_speed
        if self.scenario_def and self.scenario_def.target_speed_override is not None:
            for comp, _ in self._base_speed_targets:
                comp.params["target_speed_ms"] = float(
                    self.scenario_def.target_speed_override)
            if hasattr(self.reward_engine, 'config'):
                self.reward_engine.config.target_speed = float(
                    self.scenario_def.target_speed_override)

        # Apply scenario obstacle overrides if defined — authored
        # obstacle_overrides AND legacy ScenarioConfig.obstacles. Scenario-
        # spawned entities are tagged so set_scenario()/reset() replaces
        # them instead of accumulating duplicates. Creation errors are
        # collected and surfaced via info['scenario_warnings'], never
        # silently dropped.
        self._scenario_warnings = []
        _obs_lists = []
        if self.scenario_def and self.scenario_def.obstacle_overrides:
            _obs_lists.append(self.scenario_def.obstacle_overrides)
        if getattr(self.scenario, "obstacles", None):
            _obs_lists.append(self.scenario.obstacles)
        for obs_list in _obs_lists:
            for obs_data in obs_list:
                try:
                    from sim_core.world.entity import create_entity
                    ent_type = obs_data.get("entity_type", "cone")
                    pos_raw = obs_data.get("pos", [0.0, 0.0, 0.0])
                    ent_pos = Vec3(pos_raw[0], pos_raw[1], pos_raw[2] if len(pos_raw) > 2 else 0.0)
                    ent_yaw = float(obs_data.get("yaw", 0.0))
                    ent = create_entity(ent_type, pos=ent_pos, yaw=ent_yaw)
                    ent.name = obs_data.get("name", ent.name)
                    ent._scenario_spawned = True
                    self.add_entity(ent)
                except Exception as e:
                    self._scenario_warnings.append(
                        f"scenario entity '{obs_data.get('name', ent_type)}' "
                        f"failed to spawn: {e}")

        # Reset trackers and engines
        self.checkpoint_tracker.reset(start_time=0.0)
        self.reward_engine.reset(initial_s=0.0)
        self.termination_engine.reset()
        if self.compiled_reward_engine:
            self.compiled_reward_engine.reset(initial_s=0.0)
        if self.compiled_termination_evaluator:
            self.compiled_termination_evaluator.reset()
        if self.compiled_action_decoder:
            self.compiled_action_decoder.reset()

        self.sensors.reset_all()

        # Update initial sensor readings
        context = self._build_sensor_context()
        self.sensors.update_all(self.clock.sim_time, context, self.clock.rng)

        # Construct initial observation
        obs = self._build_observation()
        info = self._build_info_dict(terminated=False, truncated=False, reason="reset")

        return obs, info

    def step(self, action: Union[np.ndarray, List[float], int, float, Any]) -> Tuple[Union[np.ndarray, Dict[str, Any]], float, bool, bool, Dict[str, Any]]:
        """
        Executes one environment step.
        Returns: (obs, reward, terminated, truncated, info)
        """
        if self.is_done:
            last_obs = self._build_observation()
            info = self._build_info_dict(
                terminated=True,
                truncated=True,
                reason="invalid_call_after_done",
                is_colliding=self.vehicle.state.is_colliding
            )
            info['error'] = "step() was called after episode ended. Must call reset() before stepping."
            return last_obs, 0.0, True, True, info

        self.current_step += 1
        dt = self.clock.advance_fixed_step()

        # 1. Action Validation & Decoding
        if self.compiled_action_decoder:
            action_valid, action_err = self.compiled_action_decoder.validate_action(action)
            decoded = self.compiled_action_decoder.decode(action, dt=dt)
            steer = float(decoded[0]) if len(decoded) > 0 else 0.0
            throttle = float(decoded[1]) if len(decoded) > 1 else 0.0
            brake = float(decoded[2]) if len(decoded) > 2 else 0.0
        else:
            action_valid, action_err = self.action_config.validate_action(action)
            steer, throttle, brake = self.action_config.decode_action(action)

        self.last_action = [steer, throttle, brake]

        # 2. Physics Step — surface properties are queried at the current
        #    position: control-point friction, curb/off-road regions, banking
        #    and grade all modulate the dynamics.
        prev_pos_2d = Vec2(self.vehicle.state.pos.x, self.vehicle.state.pos.y)
        surf = self.track_queries.query_surface(prev_pos_2d)
        step_friction = self.active_surface_friction * surf['friction_mult']
        # Gravity bias: banking tilts the road plane (authored deg, >0 = left
        # edge raised -> pull toward -normal); grade pulls along the tangent.
        g = 9.81
        n2d = surf['normal_2d']
        t2d = surf['tangent_2d']
        bank_pull = -g * math.sin(surf['banking_rad'])
        grade_pull = -g * surf['grade']
        world_bias = Vec2(bank_pull * n2d.x + grade_pull * t2d.x,
                          bank_pull * n2d.y + grade_pull * t2d.y)
        self.vehicle.step(
            steering_cmd=steer,
            throttle_cmd=throttle,
            brake_cmd=brake,
            dt=dt,
            surface_friction=step_friction,
            world_accel_bias=world_bias
        )
        curr_pos_2d = Vec2(self.vehicle.state.pos.x, self.vehicle.state.pos.y)

        # Update dynamic entities
        for ent in self.entities:
            ent.update(dt)

        # 3. Collision Checks
        bound_col = VehicleCollisionChecker.check_track_boundary_collision(
            self.vehicle,
            self.track.all_boundary_segments,
            broadphase=self.track.broadphase
        )
        obs_col = VehicleCollisionChecker.check_obstacle_collision(
            self.vehicle,
            self.obstacles,
            broadphase=self.track.broadphase
        )
        is_colliding = bound_col.collided or obs_col.collided

        # Contact response: detected contacts get a positional depenetration
        # plus an inelastic impulse with Coulomb friction — barriers are
        # physical walls, not just flags, and the response can only remove
        # kinetic energy.
        if bound_col.collided and bound_col.contact_seg is not None:
            # Interior direction: opposite the lateral-offset side, so the
            # contact resolves the car back onto the road even if its centre
            # has just crossed the wall line.
            lat = surf['lateral_offset']
            hint = Vec2(-n2d.x if lat > 0.0 else n2d.x,
                        -n2d.y if lat > 0.0 else n2d.y)
            VehicleCollisionChecker.apply_boundary_contact(
                self.vehicle, *bound_col.contact_seg, interior_hint=hint)
        if obs_col.collided and obs_col.contact_obb is not None:
            VehicleCollisionChecker.apply_obstacle_contact(
                self.vehicle, obs_col.contact_obb)
        if is_colliding:
            curr_pos_2d = Vec2(self.vehicle.state.pos.x, self.vehicle.state.pos.y)

        # 4. Track Spatial Relationship
        track_info = self.track_queries.query_vehicle_pose(curr_pos_2d, self.vehicle.state.yaw)
        is_on_road = track_info['is_on_road']
        lateral_offset = track_info['lateral_offset']
        heading_error = track_info['heading_error']
        current_s = track_info['s']
        road_width = track_info['road_width']

        # Keep the planar vehicle's reference height pinned to the road
        # surface so the rendered mesh, chase camera, and mounted sensors
        # follow track elevation instead of clipping through it.
        self.vehicle.state.pos.z = track_info['elevation']

        # 5. Checkpoint Crossing
        cp_passed, lap_completed = self.checkpoint_tracker.update(
            prev_pos=prev_pos_2d,
            curr_pos=curr_pos_2d,
            current_time=self.clock.sim_time
        )

        # 6. Reward Calculation
        if self.compiled_reward_engine:
            step_reward, reward_breakdown = self.compiled_reward_engine.compute_step_reward(
                current_s=current_s,
                track_length=self.track.spline.total_length,
                is_closed=self.road_def.is_closed,
                lateral_offset=lateral_offset,
                road_width=road_width,
                speed=self.vehicle.state.speed,
                heading_error=heading_error,
                current_steer=self.vehicle.state.steering_angle,
                is_colliding=is_colliding,
                is_on_road=is_on_road,
                checkpoint_passed=cp_passed,
                lap_completed=lap_completed,
                dt=dt
            )
            self.reward_engine.last_breakdown = reward_breakdown
            self.reward_engine.total_accumulated_reward = self.compiled_reward_engine.total_accumulated_reward
        else:
            step_reward, reward_breakdown = self.reward_engine.compute_step_reward(
                current_s=current_s,
                track_length=self.track.spline.total_length,
                is_closed=self.road_def.is_closed,
                lateral_offset=lateral_offset,
                road_width=road_width,
                speed=self.vehicle.state.speed,
                heading_error=heading_error,
                current_steer=self.vehicle.state.steering_angle,
                is_colliding=is_colliding,
                is_on_road=is_on_road,
                checkpoint_passed=cp_passed,
                lap_completed=lap_completed
            )

        # 7. Termination & Truncation Evaluation
        if self.compiled_termination_evaluator:
            terminated, truncated, term_reason_dict = self.compiled_termination_evaluator.evaluate(
                dt=dt,
                is_colliding=is_colliding,
                is_on_road=is_on_road,
                heading_error=heading_error,
                checkpoint_passed=cp_passed,
                laps_completed=self.checkpoint_tracker.laps_completed,
                is_closed=self.road_def.is_closed
            )
            term_reason = term_reason_dict.get("reason", "running")
            self.last_termination_reason_dict = term_reason_dict
        else:
            terminated, truncated, term_reason = self.termination_engine.evaluate(
                dt=dt,
                is_colliding=is_colliding,
                is_on_road=is_on_road,
                heading_error=heading_error,
                checkpoint_passed=cp_passed,
                laps_completed=self.checkpoint_tracker.laps_completed,
                is_closed=self.road_def.is_closed
            )
            self.last_termination_reason_dict = {
                "reason": term_reason,
                "step": self.current_step,
                "sim_time": round(self.clock.sim_time, 4),
                "is_truncation": truncated
            }

        # Episode duration limit: scenario time_limit_override takes precedence
        effective_max_duration = (
            self._scenario_time_limit_s
            if self._scenario_time_limit_s is not None
            else self.episode_config.max_duration_seconds
        )
        if not terminated and not truncated and effective_max_duration > 0:
            if self.clock.sim_time >= effective_max_duration:
                truncated = True
                term_reason = (
                    "scenario_time_limit_exceeded"
                    if self._scenario_time_limit_s is not None
                    else "max_duration_exceeded"
                )
                self.last_termination_reason_dict = {
                    "reason": term_reason,
                    "step": self.current_step,
                    "sim_time": round(self.clock.sim_time, 4),
                    "is_truncation": True
                }

        if terminated or truncated:
            self.is_done = True

        # 8. Sensor Updates
        context = self._build_sensor_context()
        self.sensors.update_all(self.clock.sim_time, context, self.clock.rng)

        # 9. Observation & Info Assembly
        obs = self._build_observation()
        info = self._build_info_dict(
            terminated=terminated,
            truncated=truncated,
            reason=term_reason,
            reward_breakdown=reward_breakdown,
            track_info=track_info,
            is_colliding=is_colliding,
            action_valid=action_valid,
            action_error=action_err
        )

        # Exposed for the UI layer (recording, replay, HUD) so it does not
        # need to intercept step results from every call site (including
        # externally-driven TCP steps).
        self.last_step_reward = float(step_reward)
        self.last_step_info = info

        return obs, step_reward, terminated, truncated, info

    def _build_sensor_context(self) -> Dict[str, Any]:
        obstacle_segs = []
        for obs in self.obstacles:
            obstacle_segs.extend(obs.get_boundary_segments())

        return {
            'vehicle': self.vehicle,
            'track_queries': self.track_queries,
            'checkpoint_idx': self.checkpoint_tracker.current_index,
            'obstacle_segments': obstacle_segs,
            'sim_time': self.clock.sim_time,
        }

    def _build_observation(self) -> Union[np.ndarray, Dict[str, Any]]:
        """Constructs observation according to CompiledObservationPipeline or ObservationSchema."""
        if self.compiled_obs_pipeline:
            return self.compiled_obs_pipeline.build_observation(self.sensors.get_all_samples())

        schema = self.observation_schema
        st_sensor = self.sensors.get_sensor("vehicle_state")
        st_data = st_sensor.get_last_sample() if st_sensor else {}

        lidar_sensor = self.sensors.get_sensor("lidar_rays")
        lidar_data = lidar_sensor.get_last_sample() if lidar_sensor else {}

        camera_sensor = self.sensors.get_sensor("rgb_camera")
        camera_data = camera_sensor.get_last_sample() if camera_sensor else None

        if schema.flatten_vector:
            features = []
            if schema.include_speed:
                features.append(st_data.get('speed', 0.0) / 45.0)
            if schema.include_velocity:
                features.append(st_data.get('vel_x', 0.0) / 45.0)
                features.append(st_data.get('vel_y', 0.0) / 10.0)
            if schema.include_yaw_rate:
                features.append(st_data.get('yaw_rate', 0.0) / 3.0)
            if schema.include_steering_angle:
                features.append(st_data.get('steering_angle', 0.0) / 0.6)
            if schema.include_distance_from_center:
                half_w = max(1.0, st_data.get('road_width', 12.0) * 0.5)
                features.append(st_data.get('distance_from_center', 0.0) / half_w)
            if schema.include_heading_error:
                features.append(st_data.get('heading_error', 0.0) / math.pi)
            if schema.include_distance_to_checkpoint:
                features.append(min(1.0, st_data.get('distance_to_checkpoint', 0.0) / 100.0))
            if schema.include_lidar_rays:
                if 'ranges_norm' in lidar_data and lidar_data['ranges_norm'] is not None:
                    features.extend(lidar_data['ranges_norm'].tolist())
                else:
                    features.extend([1.0] * 15)

            obs_vec = np.array(features, dtype=np.float32)

            if schema.include_camera_rgb:
                img_data = camera_data if camera_data is not None else np.zeros((84, 84, 3), dtype=np.uint8)
                if len(features) == 0:
                    return img_data
                return {
                    'vector': obs_vec,
                    'image': img_data
                }
            return obs_vec

        else:
            obs_dict: Dict[str, Any] = {}
            if schema.include_speed:
                obs_dict['speed'] = float(st_data.get('speed', 0.0))
            if schema.include_velocity:
                obs_dict['vel_body'] = np.array([st_data.get('vel_x', 0.0), st_data.get('vel_y', 0.0)], dtype=np.float32)
            if schema.include_yaw_rate:
                obs_dict['yaw_rate'] = float(st_data.get('yaw_rate', 0.0))
            if schema.include_steering_angle:
                obs_dict['steering_angle'] = float(st_data.get('steering_angle', 0.0))
            if schema.include_distance_from_center:
                obs_dict['distance_from_center'] = float(st_data.get('distance_from_center', 0.0))
            if schema.include_heading_error:
                obs_dict['heading_error'] = float(st_data.get('heading_error', 0.0))
            if schema.include_distance_to_checkpoint:
                obs_dict['distance_to_checkpoint'] = float(st_data.get('distance_to_checkpoint', 0.0))
            if schema.include_lidar_rays:
                if 'ranges_norm' in lidar_data and lidar_data['ranges_norm'] is not None:
                    obs_dict['lidar_ranges'] = lidar_data['ranges_norm']
                else:
                    obs_dict['lidar_ranges'] = np.ones(15, dtype=np.float32)
            if schema.include_camera_rgb:
                obs_dict['camera_rgb'] = camera_data if camera_data is not None else np.zeros((84, 84, 3), dtype=np.uint8)
            return obs_dict

    def _build_info_dict(
        self,
        terminated: bool,
        truncated: bool,
        reason: str,
        reward_breakdown: Optional[Dict[str, float]] = None,
        track_info: Optional[Dict[str, Any]] = None,
        is_colliding: bool = False,
        action_valid: bool = True,
        action_error: str = ""
    ) -> Dict[str, Any]:
        """Builds comprehensive telemetry and inspection info."""
        st = self.vehicle.state
        return {
            'step': self.current_step,
            'sim_time': self.clock.sim_time,
            'terminated': terminated,
            'truncated': truncated,
            'termination_reason': reason,
            'termination_info': self.last_termination_reason_dict,
            'reward_breakdown': reward_breakdown or self.reward_engine.last_breakdown,
            'total_reward': self.reward_engine.total_accumulated_reward,
            'speed': float(st.speed),
            'lateral_offset': float(track_info['lateral_offset']) if track_info else 0.0,
            'heading_error': float(track_info['heading_error']) if track_info else 0.0,
            'road_width': float(track_info['road_width']) if track_info else 12.0,
            'is_on_road': bool(track_info['is_on_road']) if track_info else True,
            'is_colliding': is_colliding,
            'action_valid': action_valid,
            'action_error': action_error,
            'checkpoints_passed': self.checkpoint_tracker.total_checkpoints_passed,
            'current_checkpoint': self.checkpoint_tracker.current_index,
            'laps_completed': self.checkpoint_tracker.laps_completed,
            'lap_progress': self.checkpoint_tracker.get_progress_fraction(),
            'last_action': self.last_action,
            'scenario_warnings': list(getattr(self, '_scenario_warnings', [])),
        }
