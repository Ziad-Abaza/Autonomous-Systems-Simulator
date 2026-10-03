"""
Comprehensive Environment Inspector architecture for the AI Simulation Studio.
Provides coherent property inspection, scene hierarchy, and real-time editing for:
- Scene Hierarchy (Track, Control Points, Spawn, Checkpoints, World Entities, Sensors)
- Track geometry & boundary profiles
- Control points (width, elevation, banking, curvature)
- Placed World Entities (Obstacles, Barriers, Cones, Traffic Signs, Traffic Lights)
- RL Reward Engine configuration & parameter validation
- Sensor Suite management & Observation Schema configuration
- Vehicle dynamics parameters
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
import pygame

from sim_core.track.road_definition import RoadDefinition, ControlPoint, SpawnPoint
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.sensors.sensor_manager import SensorManager
from sim_core.sensors.raycast_sensor import RaycastSensor
from sim_core.sensors.camera_sensor import CameraSensor
from sim_core.world.entity import (
    WorldEntity, StaticObstacle, Barrier, TrafficCone,
    TrafficSign, TrafficLight, create_entity
)
from sim_env.reward_engine import RewardEngine, RewardConfig
from sim_env.spaces import ObservationSchema


class PropertyRow:
    """Represents an editable property in the inspector."""
    def __init__(
        self,
        prop_id: str,
        label: str,
        prop_type: str,  # "float", "int", "bool", "enum", "action", "label"
        current_value: Any,
        min_val: float = -1000.0,
        max_val: float = 1000.0,
        step: float = 1.0,
        options: Optional[List[str]] = None,
        unit: str = ""
    ):
        self.prop_id = prop_id
        self.label = label
        self.prop_type = prop_type
        self.current_value = current_value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.options = options or []
        self.unit = unit


class EnvironmentInspector:
    """
    Coherent, data-driven inspector and scene hierarchy for all simulation components.
    """
    # Two-tier categorized tabs
    CATEGORY_GEO = ["SCENE", "TRACK", "POINT", "ENTITY"]
    CATEGORY_SIM = ["REWARD", "SENSORS", "OBS", "VEHICLE"]
    ALL_TABS = CATEGORY_GEO + CATEGORY_SIM

    def __init__(
        self,
        road_def: RoadDefinition,
        vehicle_config: VehicleConfig,
        sensor_manager: SensorManager,
        reward_config: Optional[RewardConfig] = None,
        observation_schema: Optional[ObservationSchema] = None,
        entities: Optional[List[WorldEntity]] = None
    ):
        self.road_def = road_def
        self.vehicle_config = vehicle_config
        self.sensor_manager = sensor_manager
        self.reward_config = reward_config or RewardConfig()
        self.observation_schema = observation_schema or ObservationSchema()
        self.entities = entities if entities is not None else []

        self.active_tab = "SCENE"
        self.selected_point_idx: Optional[int] = 0 if road_def.control_points else None
        self.selected_entity_id: Optional[str] = None
        self.selected_sensor_name: str = "lidar_rays"

        # Action callbacks
        self.on_rebuild_mesh: Optional[Callable[[], None]] = None
        self.on_save_project: Optional[Callable[[], None]] = None
        self.on_load_project: Optional[Callable[[], None]] = None
        self.on_new_project: Optional[Callable[[], None]] = None
        self.on_start_place_tool: Optional[Callable[[str], None]] = None
        self.on_select_entity_callback: Optional[Callable[[str], None]] = None
        self.on_select_point_callback: Optional[Callable[[int], None]] = None

    def select_control_point(self, idx: Optional[int]) -> None:
        self.selected_point_idx = idx
        if idx is not None:
            self.selected_entity_id = None
            self.active_tab = "POINT"

    def select_entity(self, entity_id: Optional[str]) -> None:
        self.selected_entity_id = entity_id
        if entity_id is not None:
            self.selected_point_idx = None
            self.active_tab = "ENTITY"

    def get_selected_entity(self) -> Optional[WorldEntity]:
        if self.selected_entity_id is None:
            return None
        for ent in self.entities:
            if ent.entity_id == self.selected_entity_id:
                return ent
        return None

    def get_properties_for_active_tab(self) -> List[PropertyRow]:
        props: List[PropertyRow] = []

        # 1. SCENE HIERARCHY
        if self.active_tab == "SCENE":
            props.append(PropertyRow("sc_head_geom", "--- TRACK GEOMETRY ---", "label", ""))
            props.append(PropertyRow("sc_track_info", f"Track: {self.road_def.name}", "label", f"{len(self.road_def.control_points)} pts"))
            props.append(PropertyRow("sc_spawn_info", "Spawn Point", "label", f"({self.road_def.spawn_point.x:.0f}, {self.road_def.spawn_point.y:.0f})"))
            props.append(PropertyRow("sc_cp_info", "Checkpoints", "label", f"{self.road_def.num_checkpoints} gates"))

            props.append(PropertyRow("sc_head_ent", f"--- ENTITIES ({len(self.entities)}) ---", "label", ""))
            for i, ent in enumerate(self.entities[:8]):  # Show up to 8 entities
                props.append(PropertyRow(f"sc_ent_sel_{ent.entity_id}", ent.name, "action", f"SELECT #{ent.entity_id}"))

            props.append(PropertyRow("sc_head_add", "--- PLACE NEW ENTITY ---", "label", ""))
            props.append(PropertyRow("sc_add_obstacle", "+ Box Obstacle", "action", "PLACE BOX"))
            props.append(PropertyRow("sc_add_barrier", "+ Concrete Barrier", "action", "PLACE BARRIER"))
            props.append(PropertyRow("sc_add_cone", "+ Traffic Cone", "action", "PLACE CONE"))
            props.append(PropertyRow("sc_add_sign", "+ Traffic Sign", "action", "PLACE SIGN"))
            props.append(PropertyRow("sc_add_light", "+ Traffic Light", "action", "PLACE LIGHT"))

        # 2. TRACK DEFINITION
        elif self.active_tab == "TRACK":
            r = self.road_def
            b = r.boundary_config
            props.append(PropertyRow("track_closed", "Is Closed Loop", "bool", r.is_closed))
            props.append(PropertyRow("track_friction", "Surface Friction", "float", r.default_friction, 0.1, 2.5, 0.05))
            props.append(PropertyRow("track_checkpoints", "Checkpoints", "int", r.num_checkpoints, 4, 64, 2))
            props.append(PropertyRow("b_left_type", "Left Boundary", "enum", b.left_type, options=["guardrail", "wall", "curb", "open"]))
            props.append(PropertyRow("b_right_type", "Right Boundary", "enum", b.right_type, options=["guardrail", "wall", "curb", "open"]))
            props.append(PropertyRow("b_has_curbs", "Curb Ribbons", "bool", b.has_curbs))
            props.append(PropertyRow("b_curb_width", "Curb Width", "float", b.curb_width, 0.2, 2.0, 0.1, unit="m"))
            props.append(PropertyRow("b_wall_height", "Wall Height", "float", b.wall_height, 0.3, 3.0, 0.1, unit="m"))

        # 3. CONTROL POINT
        elif self.active_tab == "POINT":
            if self.selected_point_idx is not None and 0 <= self.selected_point_idx < len(self.road_def.control_points):
                cp = self.road_def.control_points[self.selected_point_idx]
                props.append(PropertyRow("cp_idx_display", f"Point #{self.selected_point_idx}", "label", f"Total: {len(self.road_def.control_points)}"))
                props.append(PropertyRow("cp_pos_x", "Coord X", "float", cp.x, -1000.0, 1000.0, 1.0, unit="m"))
                props.append(PropertyRow("cp_pos_y", "Coord Y", "float", cp.y, -1000.0, 1000.0, 1.0, unit="m"))
                props.append(PropertyRow("cp_width", "Road Width", "float", cp.width, 4.0, 40.0, 1.0, unit="m"))
                props.append(PropertyRow("cp_elevation", "Elevation (Z)", "float", cp.z, -50.0, 100.0, 0.5, unit="m"))
                props.append(PropertyRow("cp_banking", "Banking", "float", cp.banking, -30.0, 30.0, 1.0, unit="°"))
                props.append(PropertyRow("cp_friction", "Friction Mult", "float", cp.friction, 0.1, 2.0, 0.1))
                props.append(PropertyRow("cp_action_del", "Delete Point", "action", "DELETE POINT"))
            else:
                props.append(PropertyRow("cp_none", "No Point Selected", "label", "Click point on canvas"))

        # 4. PLACED WORLD ENTITY
        elif self.active_tab == "ENTITY":
            ent = self.get_selected_entity()
            if ent:
                props.append(PropertyRow("ent_name", f"Entity: {ent.name}", "label", f"ID: {ent.entity_id}"))
                props.append(PropertyRow("ent_type", "Type / Category", "label", f"{ent.entity_type} ({ent.semantic_label})"))
                props.append(PropertyRow("ent_pos_x", "Position X", "float", ent.pos.x, -1000.0, 1000.0, 1.0, unit="m"))
                props.append(PropertyRow("ent_pos_y", "Position Y", "float", ent.pos.y, -1000.0, 1000.0, 1.0, unit="m"))
                props.append(PropertyRow("ent_pos_z", "Elevation Z", "float", ent.pos.z, -50.0, 100.0, 0.2, unit="m"))
                deg = math.degrees(ent.yaw)
                props.append(PropertyRow("ent_yaw", "Heading (Yaw)", "float", deg, -180.0, 180.0, 5.0, unit="°"))

                if isinstance(ent, (StaticObstacle, Barrier)):
                    props.append(PropertyRow("ent_len", "Length", "float", ent.length, 0.2, 20.0, 0.2, unit="m"))
                    props.append(PropertyRow("ent_wid", "Width", "float", ent.width, 0.2, 10.0, 0.1, unit="m"))
                    props.append(PropertyRow("ent_hgt", "Height", "float", ent.height, 0.2, 5.0, 0.1, unit="m"))
                    props.append(PropertyRow("ent_col", "Collidable", "bool", ent.is_collidable))

                elif isinstance(ent, TrafficCone):
                    props.append(PropertyRow("ent_rad", "Radius", "float", ent.radius, 0.1, 1.5, 0.05, unit="m"))
                    props.append(PropertyRow("ent_hgt", "Height", "float", ent.height, 0.3, 2.0, 0.05, unit="m"))

                elif isinstance(ent, TrafficSign):
                    props.append(PropertyRow("ent_sign_type", "Sign Type", "enum", ent.sign_type, options=TrafficSign.SIGN_TYPES))
                    props.append(PropertyRow("ent_hgt", "Height", "float", ent.height, 1.0, 4.0, 0.2, unit="m"))

                elif isinstance(ent, TrafficLight):
                    props.append(PropertyRow("ent_light_state", "State", "enum", ent.state, options=TrafficLight.STATES))
                    props.append(PropertyRow("ent_green_t", "Green Time", "float", ent.green_duration, 2.0, 60.0, 1.0, unit="s"))
                    props.append(PropertyRow("ent_yellow_t", "Yellow Time", "float", ent.yellow_duration, 1.0, 10.0, 0.5, unit="s"))
                    props.append(PropertyRow("ent_red_t", "Red Time", "float", ent.red_duration, 2.0, 60.0, 1.0, unit="s"))

                props.append(PropertyRow("ent_del", "Delete Entity", "action", "DELETE ENTITY"))
            else:
                props.append(PropertyRow("ent_none", "No Entity Selected", "label", "Click entity on canvas or in scene"))

        # 5. REWARDS CONFIGURATION (Phase 2B)
        elif self.active_tab == "REWARD":
            rc = self.reward_config
            props.append(PropertyRow("rc_progress", "Progress Weight", "float", rc.weight_progress, 0.0, 10.0, 0.1))
            props.append(PropertyRow("rc_centering", "Centering Weight", "float", rc.weight_centering, 0.0, 5.0, 0.1))
            props.append(PropertyRow("rc_speed", "Speed Weight", "float", rc.weight_speed, 0.0, 5.0, 0.05))
            props.append(PropertyRow("rc_target_speed", "Target Speed", "float", rc.target_speed, 5.0, 60.0, 2.0, unit="m/s"))
            props.append(PropertyRow("rc_heading", "Heading Weight", "float", rc.weight_heading, 0.0, 5.0, 0.05))
            props.append(PropertyRow("rc_smoothness", "Smoothness Penalty", "float", rc.weight_action_smoothness, 0.0, 1.0, 0.01))
            props.append(PropertyRow("rc_checkpoint", "Checkpoint Bonus", "float", rc.checkpoint_bonus, 0.0, 100.0, 5.0))
            props.append(PropertyRow("rc_lap", "Lap Bonus", "float", rc.lap_completion_bonus, 0.0, 500.0, 25.0))
            props.append(PropertyRow("rc_collision", "Collision Penalty", "float", rc.collision_penalty, 0.0, 200.0, 10.0))
            props.append(PropertyRow("rc_off_road", "Off-Road Penalty", "float", rc.off_road_penalty, 0.0, 100.0, 5.0))
            props.append(PropertyRow("rc_backward", "Backward Penalty", "float", rc.backward_penalty, 0.0, 10.0, 0.5))

        # 6. SENSORS SUITE (Phase 2C)
        elif self.active_tab == "SENSORS":
            sensor_names = list(self.sensor_manager.sensors.keys())
            props.append(PropertyRow("sn_select", "Active Sensor", "enum", self.selected_sensor_name, options=sensor_names))
            s = self.sensor_manager.get_sensor(self.selected_sensor_name)
            if s:
                props.append(PropertyRow("sn_type", "Sensor Type", "label", s.sensor_type))
                props.append(PropertyRow("sn_freq", "Frequency", "float", s.update_frequency_hz, 5.0, 120.0, 5.0, unit="Hz"))
                props.append(PropertyRow("sn_noise", "Noise Std", "float", s.noise_std, 0.0, 2.0, 0.01))
                props.append(PropertyRow("sn_latency", "Latency", "float", s.latency_seconds, 0.0, 0.5, 0.02, unit="s"))
                if isinstance(s, RaycastSensor):
                    props.append(PropertyRow("sn_rays", "Num Rays", "int", s.num_rays, 5, 45, 2))
                    props.append(PropertyRow("sn_range", "Max Range", "float", s.max_range, 10.0, 100.0, 5.0, unit="m"))
                    props.append(PropertyRow("sn_fov", "LiDAR FOV", "float", s.fov_degrees, 30.0, 360.0, 15.0, unit="°"))
                elif isinstance(s, CameraSensor):
                    props.append(PropertyRow("sn_cam_res", f"Resolution ({s.width}x{s.height})", "label", "84x84 RGB"))
                    props.append(PropertyRow("sn_cam_fov", "Camera FOV", "float", s.fov_degrees, 40.0, 120.0, 5.0, unit="°"))
                    props.append(PropertyRow("sn_cam_pitch", "Pitch Angle", "float", math.degrees(s.local_pitch), -30.0, 15.0, 1.0, unit="°"))

        # 7. OBSERVATION SCHEMA (Phase 2C)
        elif self.active_tab == "OBS":
            obs = self.observation_schema
            props.append(PropertyRow("obs_dim", "Vector Dim", "label", f"{obs.compute_vector_dim()} features"))
            props.append(PropertyRow("obs_flatten", "Flatten to 1D Box", "bool", obs.flatten_vector))
            props.append(PropertyRow("obs_speed", "Speed (scalar)", "bool", obs.include_speed))
            props.append(PropertyRow("obs_vel", "Body Velocity (vx, vy)", "bool", obs.include_velocity))
            props.append(PropertyRow("obs_yaw_rate", "Yaw Rate", "bool", obs.include_yaw_rate))
            props.append(PropertyRow("obs_steer", "Steering Angle", "bool", obs.include_steering_angle))
            props.append(PropertyRow("obs_center", "Lateral Deviation", "bool", obs.include_distance_from_center))
            props.append(PropertyRow("obs_heading", "Heading Error", "bool", obs.include_heading_error))
            props.append(PropertyRow("obs_checkpoint", "Distance to CP", "bool", obs.include_distance_to_checkpoint))
            props.append(PropertyRow("obs_lidar", "LiDAR Rays [15]", "bool", obs.include_lidar_rays))
            props.append(PropertyRow("obs_camera", "Synthetic RGB Camera", "bool", obs.include_camera_rgb))

        # 8. VEHICLE CONFIG
        elif self.active_tab == "VEHICLE":
            vc = self.vehicle_config
            props.append(PropertyRow("vc_mass", "Vehicle Mass", "float", vc.mass, 500.0, 4000.0, 50.0, unit="kg"))
            props.append(PropertyRow("vc_wheelbase", "Wheelbase", "float", vc.wheelbase, 1.5, 4.5, 0.1, unit="m"))
            props.append(PropertyRow("vc_drive_force", "Max Drive Force", "float", vc.max_drive_force, 1000.0, 15000.0, 500.0, unit="N"))
            props.append(PropertyRow("vc_top_speed", "Top Speed", "float", vc.top_speed, 10.0, 80.0, 2.0, unit="m/s"))
            props.append(PropertyRow("vc_steer_rate", "Steer Rate", "float", vc.steering_rate, 0.5, 8.0, 0.5, unit="rad/s"))
            props.append(PropertyRow("vc_friction", "Tire Friction", "float", vc.tire_friction, 0.2, 2.5, 0.1))

        # 9. SPAWN POINT
        elif self.active_tab == "SPAWN":
            sp = self.road_def.spawn_point
            props.append(PropertyRow("sp_pos_x", "Position X", "float", sp.x, -500.0, 500.0, 1.0, unit="m"))
            props.append(PropertyRow("sp_pos_y", "Position Y", "float", sp.y, -500.0, 500.0, 1.0, unit="m"))
            props.append(PropertyRow("sp_elevation", "Elevation Z", "float", sp.z, -50.0, 100.0, 0.2, unit="m"))
            props.append(PropertyRow("sp_yaw", "Heading (Yaw)", "float", math.degrees(sp.yaw), -180.0, 180.0, 5.0, unit="°"))
            props.append(PropertyRow("sp_speed", "Initial Speed", "float", sp.initial_speed, 0.0, 60.0, 2.0, unit="m/s"))
            props.append(PropertyRow("sp_action_place", "Click Canvas to Place", "action", "PLACE SPAWN"))

        return props


    def handle_property_change(self, prop_id: str, delta_or_value: Any) -> None:
        """Applies property updates directly to underlying data models with validation."""
        r = self.road_def
        b = r.boundary_config
        vc = self.vehicle_config
        rc = self.reward_config
        obs = self.observation_schema

        # 1. Track
        if prop_id == "track_closed":
            r.is_closed = not r.is_closed
        elif prop_id == "track_friction":
            r.default_friction = round(max(0.1, min(2.5, r.default_friction + delta_or_value)), 2)
        elif prop_id == "track_checkpoints":
            r.num_checkpoints = int(max(4, min(64, r.num_checkpoints + delta_or_value)))
        elif prop_id == "b_left_type":
            opts = ["guardrail", "wall", "curb", "open"]
            idx = (opts.index(b.left_type) + 1) % len(opts)
            b.left_type = opts[idx]
        elif prop_id == "b_right_type":
            opts = ["guardrail", "wall", "curb", "open"]
            idx = (opts.index(b.right_type) + 1) % len(opts)
            b.right_type = opts[idx]
        elif prop_id == "b_has_curbs":
            b.has_curbs = not b.has_curbs
        elif prop_id == "b_curb_width":
            b.curb_width = round(max(0.2, min(2.0, b.curb_width + delta_or_value)), 2)
        elif prop_id == "b_wall_height":
            b.wall_height = round(max(0.3, min(3.0, b.wall_height + delta_or_value)), 2)

        # 2. Control Point
        elif prop_id.startswith("cp_"):
            if self.selected_point_idx is not None and 0 <= self.selected_point_idx < len(r.control_points):
                cp = r.control_points[self.selected_point_idx]
                if prop_id == "cp_pos_x":
                    cp.x = round(cp.x + delta_or_value, 1)
                elif prop_id == "cp_pos_y":
                    cp.y = round(cp.y + delta_or_value, 1)
                elif prop_id == "cp_width":
                    cp.width = round(max(4.0, min(40.0, cp.width + delta_or_value)), 1)
                elif prop_id == "cp_elevation":
                    cp.z = round(cp.z + delta_or_value, 2)
                elif prop_id == "cp_banking":
                    cp.banking = round(max(-30.0, min(30.0, cp.banking + delta_or_value)), 1)
                elif prop_id == "cp_friction":
                    cp.friction = round(max(0.1, min(2.0, cp.friction + delta_or_value)), 2)
                elif prop_id == "cp_action_del":
                    if len(r.control_points) > 3:
                        r.control_points.pop(self.selected_point_idx)
                        self.selected_point_idx = max(0, self.selected_point_idx - 1)

        # 3. Spawn Point
        elif prop_id.startswith("sp_"):
            sp = r.spawn_point
            if prop_id == "sp_pos_x":
                sp.x = round(sp.x + delta_or_value, 1)
            elif prop_id == "sp_pos_y":
                sp.y = round(sp.y + delta_or_value, 1)
            elif prop_id == "sp_elevation":
                sp.z = round(sp.z + delta_or_value, 2)
            elif prop_id == "sp_yaw":
                deg = (math.degrees(sp.yaw) + delta_or_value) % 360.0
                if deg > 180.0:
                    deg -= 360.0
                sp.yaw = math.radians(deg)
            elif prop_id == "sp_speed":
                sp.initial_speed = max(0.0, min(60.0, sp.initial_speed + delta_or_value))
            elif prop_id == "sp_action_place":
                if self.on_start_place_tool:
                    self.on_start_place_tool("spawn")

        # 4. Entity
        elif prop_id.startswith("ent_"):

            ent = self.get_selected_entity()
            if ent:
                if prop_id == "ent_pos_x":
                    ent.pos.x = round(ent.pos.x + delta_or_value, 1)
                elif prop_id == "ent_pos_y":
                    ent.pos.y = round(ent.pos.y + delta_or_value, 1)
                elif prop_id == "ent_pos_z":
                    ent.pos.z = round(ent.pos.z + delta_or_value, 2)
                elif prop_id == "ent_yaw":
                    deg = (math.degrees(ent.yaw) + delta_or_value) % 360.0
                    if deg > 180.0:
                        deg -= 360.0
                    ent.yaw = math.radians(deg)
                elif prop_id == "ent_len" and hasattr(ent, 'length'):
                    ent.length = round(max(0.2, min(20.0, ent.length + delta_or_value)), 2)
                elif prop_id == "ent_wid" and hasattr(ent, 'width'):
                    ent.width = round(max(0.2, min(10.0, ent.width + delta_or_value)), 2)
                elif prop_id == "ent_hgt" and hasattr(ent, 'height'):
                    ent.height = round(max(0.2, min(5.0, ent.height + delta_or_value)), 2)
                elif prop_id == "ent_rad" and hasattr(ent, 'radius'):
                    ent.radius = round(max(0.1, min(1.5, ent.radius + delta_or_value)), 2)
                elif prop_id == "ent_col" and hasattr(ent, 'is_collidable'):
                    ent.is_collidable = not ent.is_collidable
                elif prop_id == "ent_sign_type" and isinstance(ent, TrafficSign):
                    idx = (TrafficSign.SIGN_TYPES.index(ent.sign_type) + 1) % len(TrafficSign.SIGN_TYPES)
                    ent.sign_type = TrafficSign.SIGN_TYPES[idx]
                elif prop_id == "ent_light_state" and isinstance(ent, TrafficLight):
                    idx = (TrafficLight.STATES.index(ent.state) + 1) % len(TrafficLight.STATES)
                    ent.state = TrafficLight.STATES[idx]
                elif prop_id == "ent_green_t" and isinstance(ent, TrafficLight):
                    ent.green_duration = max(2.0, min(60.0, ent.green_duration + delta_or_value))
                elif prop_id == "ent_yellow_t" and isinstance(ent, TrafficLight):
                    ent.yellow_duration = max(1.0, min(10.0, ent.yellow_duration + delta_or_value))
                elif prop_id == "ent_red_t" and isinstance(ent, TrafficLight):
                    ent.red_duration = max(2.0, min(60.0, ent.red_duration + delta_or_value))
                elif prop_id == "ent_del":
                    if ent in self.entities:
                        self.entities.remove(ent)
                    self.selected_entity_id = None

        # 4. Rewards (Phase 2B)
        elif prop_id.startswith("rc_"):
            if prop_id == "rc_progress":
                rc.weight_progress = round(max(0.0, min(10.0, rc.weight_progress + delta_or_value)), 2)
            elif prop_id == "rc_centering":
                rc.weight_centering = round(max(0.0, min(5.0, rc.weight_centering + delta_or_value)), 2)
            elif prop_id == "rc_speed":
                rc.weight_speed = round(max(0.0, min(5.0, rc.weight_speed + delta_or_value)), 2)
            elif prop_id == "rc_target_speed":
                rc.target_speed = round(max(5.0, min(60.0, rc.target_speed + delta_or_value)), 1)
            elif prop_id == "rc_heading":
                rc.weight_heading = round(max(0.0, min(5.0, rc.weight_heading + delta_or_value)), 2)
            elif prop_id == "rc_smoothness":
                rc.weight_action_smoothness = round(max(0.0, min(1.0, rc.weight_action_smoothness + delta_or_value)), 3)
            elif prop_id == "rc_checkpoint":
                rc.checkpoint_bonus = round(max(0.0, min(100.0, rc.checkpoint_bonus + delta_or_value)), 1)
            elif prop_id == "rc_lap":
                rc.lap_completion_bonus = round(max(0.0, min(500.0, rc.lap_completion_bonus + delta_or_value)), 1)
            elif prop_id == "rc_collision":
                rc.collision_penalty = round(max(0.0, min(200.0, rc.collision_penalty + delta_or_value)), 1)
            elif prop_id == "rc_off_road":
                rc.off_road_penalty = round(max(0.0, min(100.0, rc.off_road_penalty + delta_or_value)), 1)
            elif prop_id == "rc_backward":
                rc.backward_penalty = round(max(0.0, min(10.0, rc.backward_penalty + delta_or_value)), 2)

        # 5. Sensors (Phase 2C)
        elif prop_id.startswith("sn_"):
            if prop_id == "sn_select":
                sensor_names = list(self.sensor_manager.sensors.keys())
                if sensor_names:
                    idx = (sensor_names.index(self.selected_sensor_name) + 1) % len(sensor_names) if self.selected_sensor_name in sensor_names else 0
                    self.selected_sensor_name = sensor_names[idx]
            else:
                s = self.sensor_manager.get_sensor(self.selected_sensor_name)
                if s:
                    if prop_id == "sn_freq":
                        s.update_frequency_hz = max(5.0, min(120.0, s.update_frequency_hz + delta_or_value))
                        s.update_interval = 1.0 / s.update_frequency_hz
                    elif prop_id == "sn_noise":
                        s.noise_std = round(max(0.0, min(2.0, s.noise_std + delta_or_value)), 3)
                    elif prop_id == "sn_latency":
                        s.latency_seconds = round(max(0.0, min(0.5, s.latency_seconds + delta_or_value)), 3)
                    elif prop_id == "sn_rays" and isinstance(s, RaycastSensor):
                        s.num_rays = int(max(5, min(45, s.num_rays + int(delta_or_value))))
                        s._angles = s._compute_ray_angles()
                    elif prop_id == "sn_range" and isinstance(s, RaycastSensor):
                        s.max_range = max(10.0, min(100.0, s.max_range + delta_or_value))
                    elif prop_id == "sn_fov" and isinstance(s, RaycastSensor):
                        s.fov_degrees = max(30.0, min(360.0, s.fov_degrees + delta_or_value))
                        s._angles = s._compute_ray_angles()
                    elif prop_id == "sn_cam_fov" and isinstance(s, CameraSensor):
                        s.fov_degrees = max(40.0, min(120.0, s.fov_degrees + delta_or_value))
                    elif prop_id == "sn_cam_pitch" and isinstance(s, CameraSensor):
                        s.local_pitch = math.radians(round(max(-30.0, min(15.0, math.degrees(s.local_pitch) + delta_or_value)), 1))

        # 6. Observation Schema
        elif prop_id.startswith("obs_"):
            if prop_id == "obs_flatten":
                obs.flatten_vector = not obs.flatten_vector
            elif prop_id == "obs_speed":
                obs.include_speed = not obs.include_speed
            elif prop_id == "obs_vel":
                obs.include_velocity = not obs.include_velocity
            elif prop_id == "obs_yaw_rate":
                obs.include_yaw_rate = not obs.include_yaw_rate
            elif prop_id == "obs_steer":
                obs.include_steering_angle = not obs.include_steering_angle
            elif prop_id == "obs_center":
                obs.include_distance_from_center = not obs.include_distance_from_center
            elif prop_id == "obs_heading":
                obs.include_heading_error = not obs.include_heading_error
            elif prop_id == "obs_checkpoint":
                obs.include_distance_to_checkpoint = not obs.include_distance_to_checkpoint
            elif prop_id == "obs_lidar":
                obs.include_lidar_rays = not obs.include_lidar_rays
            elif prop_id == "obs_camera":
                obs.include_camera_rgb = not obs.include_camera_rgb

        # 7. Vehicle
        elif prop_id.startswith("vc_"):
            if prop_id == "vc_mass":
                vc.mass = max(500.0, min(4000.0, vc.mass + delta_or_value))
            elif prop_id == "vc_wheelbase":
                vc.wheelbase = round(max(1.5, min(4.5, vc.wheelbase + delta_or_value)), 2)
            elif prop_id == "vc_drive_force":
                vc.max_drive_force = max(1000.0, min(15000.0, vc.max_drive_force + delta_or_value))
            elif prop_id == "vc_top_speed":
                vc.top_speed = max(10.0, min(80.0, vc.top_speed + delta_or_value))
            elif prop_id == "vc_steer_rate":
                vc.steering_rate = round(max(0.5, min(8.0, vc.steering_rate + delta_or_value)), 2)
            elif prop_id == "vc_friction":
                vc.tire_friction = round(max(0.2, min(2.5, vc.tire_friction + delta_or_value)), 2)

    def draw(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
        fonts: Dict[str, pygame.font.Font]
    ) -> List[Tuple[pygame.Rect, str]]:
        """
        Renders the complete 2-tier Environment Inspector panel with tabs,
        property rows, and rebuild/save buttons. Returns clickable [(rect, action_id), ...].
        """
        pygame.draw.rect(surface, (16, 20, 28, 240), (x, y, w, h), border_radius=8)
        pygame.draw.rect(surface, (45, 55, 75), (x, y, w, h), 1, border_radius=8)

        f_bold = fonts['bold']
        f_small = fonts['small']
        f_mono = fonts['mono']
        clickable_buttons: List[Tuple[pygame.Rect, str]] = []

        # 1. Header
        header_y = y + 8
        surface.blit(f_bold.render("ENVIRONMENT INSPECTOR", True, (0, 210, 255)), (x + 12, header_y))

        # 2. Two-tier category tab bar
        # Row 1: Geometry & Scene
        tab_y1 = y + 32
        tab_w1 = (w - 20) // len(self.CATEGORY_GEO)
        for i, tab in enumerate(self.CATEGORY_GEO):
            tab_rect = pygame.Rect(x + 10 + i * tab_w1, tab_y1, tab_w1 - 2, 22)
            is_sel = (self.active_tab == tab)
            bg = (0, 130, 220) if is_sel else (28, 35, 48)
            txt_col = (255, 255, 255) if is_sel else (150, 165, 185)
            pygame.draw.rect(surface, bg, tab_rect, border_radius=3)
            lbl = f_small.render(tab, True, txt_col)
            surface.blit(lbl, (tab_rect.centerx - lbl.get_width() // 2, tab_rect.centery - lbl.get_height() // 2))
            clickable_buttons.append((tab_rect, f"tab_{tab}"))

        # Row 2: Simulation & RL
        tab_y2 = tab_y1 + 24
        tab_w2 = (w - 20) // len(self.CATEGORY_SIM)
        for i, tab in enumerate(self.CATEGORY_SIM):
            tab_rect = pygame.Rect(x + 10 + i * tab_w2, tab_y2, tab_w2 - 2, 22)
            is_sel = (self.active_tab == tab)
            bg = (0, 130, 220) if is_sel else (28, 35, 48)
            txt_col = (255, 255, 255) if is_sel else (150, 165, 185)
            pygame.draw.rect(surface, bg, tab_rect, border_radius=3)
            lbl = f_small.render(tab, True, txt_col)
            surface.blit(lbl, (tab_rect.centerx - lbl.get_width() // 2, tab_rect.centery - lbl.get_height() // 2))
            clickable_buttons.append((tab_rect, f"tab_{tab}"))

        # Separator line
        sep_y = tab_y2 + 26
        pygame.draw.line(surface, (45, 55, 75), (x + 10, sep_y), (x + w - 10, sep_y), 1)

        # 3. Property Rows
        props = self.get_properties_for_active_tab()
        row_y = sep_y + 6
        row_h = 26

        for p in props:
            # Check if within vertical bounds of inspector
            if row_y > (y + h - 85):
                break

            # Label
            lbl_color = (0, 200, 255) if "---" in p.label else (180, 195, 210)
            surface.blit(f_small.render(p.label, True, lbl_color), (x + 12, row_y + 4))

            if p.prop_type in ("float", "int"):
                val_str = f"{p.current_value:.1f}{p.unit}" if p.prop_type == "float" else f"{int(p.current_value)}{p.unit}"
                lbl_val = f_mono.render(val_str, True, (255, 255, 255))
                surface.blit(lbl_val, (x + w - 105 - lbl_val.get_width(), row_y + 4))

                # [-] and [+] buttons
                btn_minus = pygame.Rect(x + w - 85, row_y + 2, 35, 20)
                btn_plus = pygame.Rect(x + w - 46, row_y + 2, 35, 20)

                pygame.draw.rect(surface, (35, 45, 60), btn_minus, border_radius=3)
                pygame.draw.rect(surface, (35, 45, 60), btn_plus, border_radius=3)

                surface.blit(f_bold.render("-", True, (220, 220, 220)), (btn_minus.centerx - 4, btn_minus.centery - 7))
                surface.blit(f_bold.render("+", True, (220, 220, 220)), (btn_plus.centerx - 5, btn_plus.centery - 7))

                clickable_buttons.append((btn_minus, f"prop_minus_{p.prop_id}"))
                clickable_buttons.append((btn_plus, f"prop_plus_{p.prop_id}"))

            elif p.prop_type == "bool":
                btn_toggle = pygame.Rect(x + w - 75, row_y + 2, 65, 20)
                bg_col = (30, 140, 70) if p.current_value else (80, 40, 40)
                txt = "TRUE" if p.current_value else "FALSE"
                pygame.draw.rect(surface, bg_col, btn_toggle, border_radius=3)
                lbl_t = f_small.render(txt, True, (255, 255, 255))
                surface.blit(lbl_t, (btn_toggle.centerx - lbl_t.get_width() // 2, btn_toggle.centery - lbl_t.get_height() // 2))
                clickable_buttons.append((btn_toggle, f"prop_toggle_{p.prop_id}"))

            elif p.prop_type == "enum":
                btn_enum = pygame.Rect(x + w - 105, row_y + 2, 95, 20)
                pygame.draw.rect(surface, (35, 48, 68), btn_enum, border_radius=3)
                lbl_e = f_small.render(str(p.current_value).upper(), True, (0, 210, 255))
                surface.blit(lbl_e, (btn_enum.centerx - lbl_e.get_width() // 2, btn_enum.centery - lbl_e.get_height() // 2))
                clickable_buttons.append((btn_enum, f"prop_enum_{p.prop_id}"))

            elif p.prop_type == "action":
                btn_act = pygame.Rect(x + w - 130, row_y + 2, 120, 20)
                color = (150, 40, 40) if "DEL" in p.prop_id else (0, 110, 170)
                pygame.draw.rect(surface, color, btn_act, border_radius=3)
                lbl_a = f_small.render(str(p.current_value), True, (255, 255, 255))
                surface.blit(lbl_a, (btn_act.centerx - lbl_a.get_width() // 2, btn_act.centery - lbl_a.get_height() // 2))
                clickable_buttons.append((btn_act, f"prop_act_{p.prop_id}"))

            elif p.prop_type == "label":
                lbl_v = f_small.render(str(p.current_value), True, (150, 160, 175))
                surface.blit(lbl_v, (x + w - 12 - lbl_v.get_width(), row_y + 4))

            row_y += row_h

        # 4. Bottom Action Buttons (Rebuild 3D, New, Save, Load)
        bot_y = y + h - 80
        pygame.draw.line(surface, (45, 55, 75), (x + 10, bot_y - 6), (x + w - 10, bot_y - 6), 1)

        # Rebuild 3D Mesh
        btn_rebuild = pygame.Rect(x + 12, bot_y, w - 24, 30)
        pygame.draw.rect(surface, (0, 140, 90), btn_rebuild, border_radius=4)
        lbl_rb = f_bold.render("REBUILD 3D ENVIRONMENT", True, (255, 255, 255))
        surface.blit(lbl_rb, (btn_rebuild.centerx - lbl_rb.get_width() // 2, btn_rebuild.centery - lbl_rb.get_height() // 2))
        clickable_buttons.append((btn_rebuild, "action_rebuild_mesh"))

        # Save & Load & New Project Row
        sub_y = bot_y + 36
        sub_w = (w - 24 - 10) // 3
        b_new = pygame.Rect(x + 12, sub_y, sub_w, 26)
        b_save = pygame.Rect(x + 12 + sub_w + 5, sub_y, sub_w, 26)
        b_load = pygame.Rect(x + 12 + (sub_w + 5) * 2, sub_y, sub_w, 26)

        pygame.draw.rect(surface, (35, 45, 60), b_new, border_radius=3)
        pygame.draw.rect(surface, (30, 90, 150), b_save, border_radius=3)
        pygame.draw.rect(surface, (45, 55, 75), b_load, border_radius=3)

        lbl_new = f_small.render("NEW", True, (230, 230, 230))
        lbl_save = f_small.render("SAVE", True, (255, 255, 255))
        lbl_load = f_small.render("LOAD", True, (230, 230, 230))

        surface.blit(lbl_new, (b_new.centerx - lbl_new.get_width() // 2, b_new.centery - lbl_new.get_height() // 2))
        surface.blit(lbl_save, (b_save.centerx - lbl_save.get_width() // 2, b_save.centery - lbl_save.get_height() // 2))
        surface.blit(lbl_load, (b_load.centerx - lbl_load.get_width() // 2, b_load.centery - lbl_load.get_height() // 2))

        clickable_buttons.append((b_new, "action_new_project"))
        clickable_buttons.append((b_save, "action_save_project"))
        clickable_buttons.append((b_load, "action_load_project"))

        return clickable_buttons
