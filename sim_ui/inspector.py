"""
Data-driven Environment Inspector architecture for the AI Simulation Studio.
Provides coherent property inspection and real-time editing for:
- Track geometry & boundaries
- Spline control points
- Spawn configuration
- Vehicle dynamic parameters
- Sensor suites
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
import pygame

from sim_core.track.road_definition import RoadDefinition, ControlPoint, SpawnPoint
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.sensors.sensor_manager import SensorManager


class PropertyRow:
    """Represents an editable property in the inspector."""
    def __init__(
        self,
        prop_id: str,
        label: str,
        prop_type: str,  # "float", "int", "bool", "enum", "action"
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
    Coherent, reusable property inspector for all simulation entities.
    """
    TABS = ["TRACK", "POINT", "SPAWN", "VEHICLE", "SENSORS"]

    def __init__(
        self,
        road_def: RoadDefinition,
        vehicle_config: VehicleConfig,
        sensor_manager: SensorManager
    ):
        self.road_def = road_def
        self.vehicle_config = vehicle_config
        self.sensor_manager = sensor_manager

        self.active_tab = "TRACK"
        self.selected_point_idx: Optional[int] = 0 if road_def.control_points else None
        self.selected_sensor_name: str = "lidar_rays"

        # Action callbacks for project management
        self.on_rebuild_mesh: Optional[Callable[[], None]] = None
        self.on_save_project: Optional[Callable[[], None]] = None
        self.on_load_project: Optional[Callable[[], None]] = None
        self.on_new_project: Optional[Callable[[], None]] = None
        self.on_start_place_spawn: Optional[Callable[[], None]] = None

    def select_control_point(self, idx: Optional[int]) -> None:
        self.selected_point_idx = idx
        if idx is not None:
            self.active_tab = "POINT"

    def get_properties_for_active_tab(self) -> List[PropertyRow]:
        props: List[PropertyRow] = []

        if self.active_tab == "TRACK":
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

        elif self.active_tab == "POINT":
            if self.selected_point_idx is not None and 0 <= self.selected_point_idx < len(self.road_def.control_points):
                cp = self.road_def.control_points[self.selected_point_idx]
                props.append(PropertyRow("cp_idx_display", f"Point #{self.selected_point_idx}", "label", f"Total: {len(self.road_def.control_points)}"))
                props.append(PropertyRow("cp_width", "Road Width", "float", cp.width, 4.0, 40.0, 1.0, unit="m"))
                props.append(PropertyRow("cp_elevation", "Elevation (Z)", "float", cp.z, -50.0, 100.0, 0.5, unit="m"))
                props.append(PropertyRow("cp_banking", "Banking", "float", cp.banking, -30.0, 30.0, 1.0, unit="°"))
                props.append(PropertyRow("cp_friction", "Friction Mult", "float", cp.friction, 0.1, 2.0, 0.1))
                props.append(PropertyRow("cp_action_del", "Delete Point", "action", "DELETE"))
            else:
                props.append(PropertyRow("cp_none", "No Point Selected", "label", "Click point on canvas"))

        elif self.active_tab == "SPAWN":
            sp = self.road_def.spawn_point
            props.append(PropertyRow("sp_pos_x", "Position X", "float", sp.x, -500.0, 500.0, 1.0, unit="m"))
            props.append(PropertyRow("sp_pos_y", "Position Y", "float", sp.y, -500.0, 500.0, 1.0, unit="m"))
            props.append(PropertyRow("sp_elevation", "Elevation Z", "float", sp.z, -50.0, 100.0, 0.2, unit="m"))
            props.append(PropertyRow("sp_yaw", "Heading (Yaw)", "float", math.degrees(sp.yaw), -180.0, 180.0, 5.0, unit="°"))
            props.append(PropertyRow("sp_speed", "Initial Speed", "float", sp.initial_speed, 0.0, 60.0, 2.0, unit="m/s"))
            props.append(PropertyRow("sp_action_place", "Click Canvas to Place", "action", "PLACE SPAWN"))

        elif self.active_tab == "VEHICLE":
            vc = self.vehicle_config
            props.append(PropertyRow("vc_mass", "Vehicle Mass", "float", vc.mass, 500.0, 4000.0, 50.0, unit="kg"))
            props.append(PropertyRow("vc_wheelbase", "Wheelbase", "float", vc.wheelbase, 1.5, 4.5, 0.1, unit="m"))
            props.append(PropertyRow("vc_drive_force", "Max Drive Force", "float", vc.max_drive_force, 1000.0, 15000.0, 500.0, unit="N"))
            props.append(PropertyRow("vc_top_speed", "Top Speed", "float", vc.top_speed, 10.0, 80.0, 2.0, unit="m/s"))
            props.append(PropertyRow("vc_steer_rate", "Steer Rate", "float", vc.steering_rate, 0.5, 8.0, 0.5, unit="rad/s"))
            props.append(PropertyRow("vc_friction", "Tire Friction", "float", vc.tire_friction, 0.2, 2.5, 0.1))

        elif self.active_tab == "SENSORS":
            sensor_names = list(self.sensor_manager.sensors.keys())
            props.append(PropertyRow("sn_select", "Select Sensor", "enum", self.selected_sensor_name, options=sensor_names))
            curr_sensor = self.sensor_manager.get_sensor(self.selected_sensor_name)
            if curr_sensor:
                props.append(PropertyRow("sn_freq", "Frequency", "float", curr_sensor.update_frequency_hz, 5.0, 120.0, 5.0, unit="Hz"))
                props.append(PropertyRow("sn_noise", "Noise Std", "float", curr_sensor.noise_std, 0.0, 2.0, 0.02))
                props.append(PropertyRow("sn_latency", "Latency", "float", curr_sensor.latency_seconds, 0.0, 0.5, 0.02, unit="s"))

        return props

    def handle_property_change(self, prop_id: str, delta_or_value: Any) -> None:
        """Applies property modifications directly to underlying models."""
        r = self.road_def
        b = r.boundary_config
        vc = self.vehicle_config

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
                if prop_id == "cp_width":
                    cp.width = round(max(4.0, min(40.0, cp.width + delta_or_value)), 1)
                elif prop_id == "cp_elevation":
                    cp.z = round(cp.z + delta_or_value, 2)
                elif prop_id == "cp_banking":
                    cp.banking = round(cp.banking + delta_or_value, 1)
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
                sp.x += delta_or_value
            elif prop_id == "sp_pos_y":
                sp.y += delta_or_value
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
                if self.on_start_place_spawn:
                    self.on_start_place_spawn()

        # 4. Vehicle Config
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

        # 5. Sensors
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
        Renders the complete Inspector window with tab navigation, property rows,
        and bottom action buttons. Returns list of clickable [(rect, action_id), ...].
        """
        # Panel background
        pygame.draw.rect(surface, (18, 22, 30, 235), (x, y, w, h), border_radius=8)
        pygame.draw.rect(surface, (45, 55, 75), (x, y, w, h), 1, border_radius=8)

        f_bold = fonts['bold']
        f_small = fonts['small']
        f_mono = fonts['mono']
        clickable_buttons: List[Tuple[pygame.Rect, str]] = []

        # 1. Header & Tabs
        header_y = y + 10
        surface.blit(f_bold.render("ENVIRONMENT INSPECTOR", True, (0, 210, 255)), (x + 12, header_y))

        tab_y = y + 36
        tab_w = (w - 20) // len(self.TABS)
        for i, tab in enumerate(self.TABS):
            tab_rect = pygame.Rect(x + 10 + i * tab_w, tab_y, tab_w - 2, 24)
            is_sel = (self.active_tab == tab)
            bg = (0, 130, 220) if is_sel else (30, 38, 50)
            txt_col = (255, 255, 255) if is_sel else (160, 175, 195)
            pygame.draw.rect(surface, bg, tab_rect, border_radius=3)
            lbl = f_small.render(tab, True, txt_col)
            surface.blit(lbl, (tab_rect.centerx - lbl.get_width() // 2, tab_rect.centery - lbl.get_height() // 2))
            clickable_buttons.append((tab_rect, f"tab_{tab}"))

        # Separator line
        pygame.draw.line(surface, (45, 55, 75), (x + 10, tab_y + 28), (x + w - 10, tab_y + 28), 1)

        # 2. Property Rows
        props = self.get_properties_for_active_tab()
        row_y = tab_y + 34
        row_h = 28

        for p in props:
            # Label
            surface.blit(f_small.render(p.label, True, (190, 200, 215)), (x + 12, row_y + 4))

            if p.prop_type in ("float", "int"):
                val_str = f"{p.current_value:.1f}{p.unit}" if p.prop_type == "float" else f"{int(p.current_value)}{p.unit}"
                lbl_val = f_mono.render(val_str, True, (255, 255, 255))
                surface.blit(lbl_val, (x + w - 110 - lbl_val.get_width(), row_y + 4))

                # [-] and [+] buttons
                btn_minus = pygame.Rect(x + w - 90, row_y + 2, 38, 22)
                btn_plus = pygame.Rect(x + w - 48, row_y + 2, 38, 22)

                pygame.draw.rect(surface, (35, 45, 60), btn_minus, border_radius=3)
                pygame.draw.rect(surface, (35, 45, 60), btn_plus, border_radius=3)

                surface.blit(f_bold.render("-", True, (220, 220, 220)), (btn_minus.centerx - 4, btn_minus.centery - 8))
                surface.blit(f_bold.render("+", True, (220, 220, 220)), (btn_plus.centerx - 5, btn_plus.centery - 8))

                clickable_buttons.append((btn_minus, f"prop_minus_{p.prop_id}"))
                clickable_buttons.append((btn_plus, f"prop_plus_{p.prop_id}"))

            elif p.prop_type == "bool":
                btn_toggle = pygame.Rect(x + w - 80, row_y + 2, 70, 22)
                bg_col = (30, 140, 70) if p.current_value else (80, 40, 40)
                txt = "TRUE" if p.current_value else "FALSE"
                pygame.draw.rect(surface, bg_col, btn_toggle, border_radius=3)
                lbl_t = f_small.render(txt, True, (255, 255, 255))
                surface.blit(lbl_t, (btn_toggle.centerx - lbl_t.get_width() // 2, btn_toggle.centery - lbl_t.get_height() // 2))
                clickable_buttons.append((btn_toggle, f"prop_toggle_{p.prop_id}"))

            elif p.prop_type == "enum":
                btn_enum = pygame.Rect(x + w - 110, row_y + 2, 100, 22)
                pygame.draw.rect(surface, (40, 50, 70), btn_enum, border_radius=3)
                lbl_e = f_small.render(str(p.current_value).upper(), True, (0, 200, 255))
                surface.blit(lbl_e, (btn_enum.centerx - lbl_e.get_width() // 2, btn_enum.centery - lbl_e.get_height() // 2))
                clickable_buttons.append((btn_enum, f"prop_enum_{p.prop_id}"))

            elif p.prop_type == "action":
                btn_act = pygame.Rect(x + 12, row_y + 2, w - 24, 24)
                color = (160, 40, 40) if "DEL" in p.prop_id else (0, 120, 180)
                pygame.draw.rect(surface, color, btn_act, border_radius=4)
                lbl_a = f_bold.render(str(p.current_value), True, (255, 255, 255))
                surface.blit(lbl_a, (btn_act.centerx - lbl_a.get_width() // 2, btn_act.centery - lbl_a.get_height() // 2))
                clickable_buttons.append((btn_act, f"prop_act_{p.prop_id}"))

            elif p.prop_type == "label":
                lbl_v = f_small.render(str(p.current_value), True, (150, 160, 175))
                surface.blit(lbl_v, (x + w - 12 - lbl_v.get_width(), row_y + 4))

            row_y += row_h

        # 3. Bottom Action Buttons (Rebuild 3D, New, Save, Load)
        bot_y = y + h - 85
        pygame.draw.line(surface, (45, 55, 75), (x + 10, bot_y - 8), (x + w - 10, bot_y - 8), 1)

        # Rebuild 3D Mesh
        btn_rebuild = pygame.Rect(x + 12, bot_y, w - 24, 32)
        pygame.draw.rect(surface, (0, 150, 100), btn_rebuild, border_radius=4)
        lbl_rb = f_bold.render("REBUILD 3D ENVIRONMENT", True, (255, 255, 255))
        surface.blit(lbl_rb, (btn_rebuild.centerx - lbl_rb.get_width() // 2, btn_rebuild.centery - lbl_rb.get_height() // 2))
        clickable_buttons.append((btn_rebuild, "action_rebuild_mesh"))

        # Save & Load & New Project Row
        sub_y = bot_y + 38
        sub_w = (w - 24 - 10) // 3
        b_new = pygame.Rect(x + 12, sub_y, sub_w, 28)
        b_save = pygame.Rect(x + 12 + sub_w + 5, sub_y, sub_w, 28)
        b_load = pygame.Rect(x + 12 + (sub_w + 5) * 2, sub_y, sub_w, 28)

        pygame.draw.rect(surface, (40, 50, 65), b_new, border_radius=3)
        pygame.draw.rect(surface, (30, 90, 150), b_save, border_radius=3)
        pygame.draw.rect(surface, (50, 60, 80), b_load, border_radius=3)

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
