"""
High-performance Gymnasium-compliant Simulation Environment.
Integrates world physics, modular vehicle dynamics, sensors, reward engine,
termination logic, and scenario conditions.
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

from sim_env.spaces import ActionSpaceConfig, ObservationSchema, ActionSpaceType
from sim_env.reward_engine import RewardEngine, RewardConfig
from sim_env.termination_engine import TerminationEngine, TerminationConfig
from sim_env.domain_randomizer import DomainRandomizer, DomainRandomizationConfig
from sim_env.scenarios import ScenarioConfig


class SimulationEnvironment:
    """
    Core AI Environment integrating Simulation + Agent Sensors + Rewards + Termination.
    Follows standard Gymnasium reset/step lifecycle.
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
        self.obstacles: List[Obstacle] = []
        self.checkpoint_tracker = CheckpointTracker(self.track)

        # Sensors
        self.sensors = sensor_manager or SensorManager.create_default_sensor_suite()

        # RL Schemas & Engines
        self.action_config = action_config or ActionSpaceConfig()
        self.observation_schema = observation_schema or ObservationSchema()
        self.reward_engine = RewardEngine(reward_config or RewardConfig())
        self.termination_engine = TerminationEngine(termination_config or TerminationConfig())
        self.domain_randomizer = DomainRandomizer(randomization_config or DomainRandomizationConfig())
        self.scenario = scenario_config or ScenarioConfig()

        # Active physical properties (modified by scenario & domain randomization)
        self.active_surface_friction = self.road_def.default_friction * self.scenario.surface_friction_mult

        # History tracking
        self.current_step = 0
        self.is_done = False
        self.last_action = [0.0, 0.0, 0.0]

    def set_road_definition(self, road_def: RoadDefinition) -> None:
        """Updates road definition and regenerates 3D mesh and collision bounds."""
        self.road_def = road_def
        self.track = TrackMeshGenerator.generate(self.road_def)
        self.track_queries = TrackSpatialQueries(self.track)
        self.checkpoint_tracker = CheckpointTracker(self.track)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def clear_obstacles(self) -> None:
        self.obstacles.clear()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, Dict[str, Any]], Dict[str, Any]]:
        """
        Resets environment to initial state, returning (initial_observation, info).
        """
        if seed is not None:
            self.clock.reset(seed=seed)
        else:
            self.clock.reset()

        self.current_step = 0
        self.is_done = False
        self.last_action = [0.0, 0.0, 0.0]

        # Domain Randomization
        dr_params = self.domain_randomizer.sample_parameters(self.clock.rng)
        self.active_surface_friction = (
            self.road_def.default_friction *
            self.scenario.surface_friction_mult *
            dr_params['surface_friction_factor']
        )

        # Configure vehicle randomized mass
        self.vehicle.config.mass = self.base_vehicle_config.mass * dr_params['mass_factor']
        self.vehicle.config.tire_friction = self.base_vehicle_config.tire_friction * dr_params['tire_friction_factor']
        self.vehicle._recompute_inertials()

        # Compute initial spawn pose
        sp = self.road_def.spawn_point
        spawn_yaw = sp.yaw + dr_params['spawn_heading_jitter']
        # Spawn jitter across road normal
        cos_y = math.cos(spawn_yaw)
        sin_y = math.sin(spawn_yaw)
        norm_x = -sin_y
        norm_y = cos_y
        spawn_x = sp.x + norm_x * dr_params['spawn_lateral_jitter']
        spawn_y = sp.y + norm_y * dr_params['spawn_lateral_jitter']

        # Reset vehicle
        self.vehicle.reset(
            pos=Vec3(spawn_x, spawn_y, sp.z),
            yaw=spawn_yaw,
            initial_speed=sp.initial_speed
        )

        # Reset trackers and engines
        self.checkpoint_tracker.reset(start_time=0.0)
        self.reward_engine.reset(initial_s=0.0)
        self.termination_engine.reset()
        self.sensors.reset_all()

        # Update initial sensor readings
        context = self._build_sensor_context()
        self.sensors.update_all(self.clock.sim_time, context, self.clock.rng)

        # Construct initial observation
        obs = self._build_observation()
        info = self._build_info_dict(terminated=False, truncated=False, reason="reset")

        return obs, info

    def step(self, action: Union[np.ndarray, List[float], int, float]) -> Tuple[Union[np.ndarray, Dict[str, Any]], float, bool, bool, Dict[str, Any]]:
        """
        Executes one environment step.
        Returns: (obs, reward, terminated, truncated, info)
        """
        if self.is_done:
            # If called after episode ended, reset automatically or return current state
            return self.reset()[0], 0.0, True, False, {"warning": "Called step after episode ended"}

        self.current_step += 1
        dt = self.clock.advance_fixed_step()

        # 1. Decode Action
        steer, throttle, brake = self.action_config.decode_action(action)
        self.last_action = [steer, throttle, brake]

        # 2. Physics Step
        prev_pos_2d = Vec2(self.vehicle.state.pos.x, self.vehicle.state.pos.y)
        self.vehicle.step(
            steering_cmd=steer,
            throttle_cmd=throttle,
            brake_cmd=brake,
            dt=dt,
            surface_friction=self.active_surface_friction
        )
        curr_pos_2d = Vec2(self.vehicle.state.pos.x, self.vehicle.state.pos.y)

        # 3. Collision Checks
        # Boundary collision
        bound_col = VehicleCollisionChecker.check_track_boundary_collision(
            self.vehicle,
            self.track.all_boundary_segments
        )
        # Obstacle collision
        obs_col = VehicleCollisionChecker.check_obstacle_collision(
            self.vehicle,
            self.obstacles
        )
        is_colliding = bound_col.collided or obs_col.collided

        # 4. Track Spatial Relationship
        track_info = self.track_queries.query_vehicle_pose(curr_pos_2d, self.vehicle.state.yaw)
        is_on_road = track_info['is_on_road']
        lateral_offset = track_info['lateral_offset']
        heading_error = track_info['heading_error']
        current_s = track_info['s']
        road_width = track_info['road_width']

        # 5. Checkpoint Crossing
        cp_passed, lap_completed = self.checkpoint_tracker.update(
            prev_pos=prev_pos_2d,
            curr_pos=curr_pos_2d,
            current_time=self.clock.sim_time
        )

        # 6. Reward Calculation & Decomposition
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
        terminated, truncated, term_reason = self.termination_engine.evaluate(
            dt=dt,
            is_colliding=is_colliding,
            is_on_road=is_on_road,
            heading_error=heading_error,
            checkpoint_passed=cp_passed,
            laps_completed=self.checkpoint_tracker.laps_completed
        )

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
            is_colliding=is_colliding
        )

        return obs, step_reward, terminated, truncated, info

    def _build_sensor_context(self) -> Dict[str, Any]:
        """Provides environment context for sensor evaluations."""
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
        """Constructs observation according to ObservationSchema."""
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
                features.append(st_data.get('speed', 0.0) / 45.0)  # Normalize ~ [0, 1]
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
            if schema.include_lidar_rays and 'ranges_norm' in lidar_data:
                features.extend(lidar_data['ranges_norm'].tolist())

            obs_vec = np.array(features, dtype=np.float32)

            if schema.include_camera_rgb and camera_data is not None:
                return {
                    'vector': obs_vec,
                    'image': camera_data
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
            if schema.include_lidar_rays and 'ranges_norm' in lidar_data:
                obs_dict['lidar_ranges'] = lidar_data['ranges_norm']
            if schema.include_camera_rgb and camera_data is not None:
                obs_dict['camera_rgb'] = camera_data
            return obs_dict

    def _build_info_dict(
        self,
        terminated: bool,
        truncated: bool,
        reason: str,
        reward_breakdown: Optional[Dict[str, float]] = None,
        track_info: Optional[Dict[str, Any]] = None,
        is_colliding: bool = False
    ) -> Dict[str, Any]:
        """Builds comprehensive telemetry and inspection info."""
        st = self.vehicle.state
        return {
            'step': self.current_step,
            'sim_time': self.clock.sim_time,
            'terminated': terminated,
            'truncated': truncated,
            'termination_reason': reason,
            'reward_breakdown': reward_breakdown or self.reward_engine.last_breakdown,
            'total_reward': self.reward_engine.total_accumulated_reward,
            'speed': float(st.speed),
            'lateral_offset': float(track_info['lateral_offset']) if track_info else 0.0,
            'heading_error': float(track_info['heading_error']) if track_info else 0.0,
            'is_on_road': bool(track_info['is_on_road']) if track_info else True,
            'is_colliding': is_colliding,
            'checkpoints_passed': self.checkpoint_tracker.total_checkpoints_passed,
            'current_checkpoint': self.checkpoint_tracker.current_index,
            'laps_completed': self.checkpoint_tracker.laps_completed,
            'lap_progress': self.checkpoint_tracker.get_progress_fraction(),
            'last_action': self.last_action,
        }
