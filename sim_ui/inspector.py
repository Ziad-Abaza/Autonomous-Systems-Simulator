"""
Comprehensive RL Environment Designer & Inspector Architecture.
Provides interactive visual authoring, scene hierarchy, and validation for:
- Environment Overview & Training Exporter
- Agent Definition & Sensor Bindings
- Observation Space Designer (Channels, normalizations, leakage validation)
- Action Space Designer (Continuous/discrete, deadzones, rate limits)
- Reward Function Designer (Components, weights, falloffs, penalties)
- Termination Designer (Terminated vs Truncated rules, thresholds)
- Scenario & Domain Randomization Designer
- Environment Validation Panel (ERROR / WARNING / INFO gating)
- Track Geometry, Control Points, and Placed World Entities
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
from sim_env.agent import AgentDefinition
from sim_env.scenario_designer import ScenarioDefinition
from sim_env.validator import EnvironmentValidator, ValidationReport
from sim_ui import theme as T


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
    Coherent, data-driven inspector and RL environment designer.
    """
    # Two-tier categorized tabs
    CATEGORY_GEO = ["OVERVIEW", "SCENE", "TRACK", "POINT", "ENTITY"]
    CATEGORY_RL = ["AGENT", "SENSORS", "OBS", "ACTION", "REWARD", "TERM", "SCENARIO", "VALIDATE", "TRAIN"]
    ALL_TABS = CATEGORY_GEO + CATEGORY_RL

    # Tab labels — row 1 fits full words; row 2 uses readable
    # abbreviations (full names available as tooltips via TAB_TOOLTIPS).
    TAB_LABELS = {
        "OVERVIEW": "Overview", "SCENE": "Scene", "TRACK": "Track",
        "POINT": "Point", "ENTITY": "Entity",
        "AGENT": "Agent", "SENSORS": "Sen", "OBS": "Obs", "ACTION": "Act",
        "REWARD": "Rwd", "TERM": "Trm", "SCENARIO": "Scn",
        "VALIDATE": "Val", "TRAIN": "Trn",
    }
    TAB_TOOLTIPS = {
        "OBS": "Observation channels — what the agent sees",
        "ACTION": "Action space — how the agent drives",
        "REWARD": "Reward function — what the agent optimizes",
        "TERM": "Termination rules — when an episode ends",
        "SCENARIO": "Scenario — weather, traffic, conditions",
        "VALIDATE": "Environment validation — training readiness gate",
        "TRAIN": "Training & experiments",
        "AGENT": "Agent definition & vehicle",
        "SENSORS": "Sensors attached to the vehicle — cameras, LiDAR, IMU",
    }
    # Validation issue subsystem → owning tab (click-to-navigate)
    SUBSYSTEM_TAB = {
        "road": "TRACK", "track": "TRACK", "geometry": "TRACK",
        "spawn": "TRACK", "checkpoint": "TRACK",
        "obs": "OBS", "sensor": "OBS", "observation": "OBS",
        "action": "ACTION", "reward": "REWARD", "term": "TERM",
        "termination": "TERM", "scenario": "SCENARIO", "agent": "AGENT",
        "entity": "ENTITY", "entities": "ENTITY",
    }

    def __init__(
        self,
        road_def: RoadDefinition,
        vehicle_config: VehicleConfig,
        sensor_manager: SensorManager,
        reward_config: Optional[RewardConfig] = None,
        observation_schema: Optional[ObservationSchema] = None,
        entities: Optional[List[WorldEntity]] = None,
        agent: Optional[AgentDefinition] = None,
        scenario_def: Optional[ScenarioDefinition] = None
    ):
        self.road_def = road_def
        self.vehicle_config = vehicle_config
        self.sensor_manager = sensor_manager
        self.reward_config = reward_config or RewardConfig()
        self.observation_schema = observation_schema or ObservationSchema()
        self.entities = entities if entities is not None else []
        self.agent: AgentDefinition = agent or AgentDefinition.create_default_vehicle_agent()
        self.scenario_def: ScenarioDefinition = scenario_def or ScenarioDefinition.get_standard_scenarios()["basic_lane_following"]

        self.active_tab = "OVERVIEW"
        self.selected_point_idx: Optional[int] = 0 if road_def.control_points else None
        self.selected_entity_id: Optional[str] = None
        self.selected_sensor_name: str = "lidar_rays"

        # Cached validation report
        self.last_validation_report: Optional[ValidationReport] = None
        self.run_validation()

        # Action callbacks
        self.on_rebuild_mesh: Optional[Callable[[], None]] = None
        self.on_save_project: Optional[Callable[[], None]] = None
        self.on_load_project: Optional[Callable[[], None]] = None
        self.on_new_project: Optional[Callable[[], None]] = None
        self.on_export_training: Optional[Callable[[], None]] = None
        self.on_load_template: Optional[Callable[[str], None]] = None
        self.on_start_place_tool: Optional[Callable[[str], None]] = None
        self.on_select_entity_callback: Optional[Callable[[str], None]] = None
        self.on_select_point_callback: Optional[Callable[[int], None]] = None
        self.on_agent_modified: Optional[Callable[[], None]] = None
        # Training & Experiments panel hooks (provided by the app layer)
        self.train_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self.on_train_action: Optional[Callable[[str], None]] = None

        # Per-tab scroll offsets + geometry of the scrollable props area
        self.tab_scroll: Dict[str, int] = {}
        self.props_area: Optional[pygame.Rect] = None
        self._props_count: int = 0

    def scroll(self, dy_steps: int) -> None:
        """Mouse-wheel scroll for the property area (called when the
        cursor is over the panel)."""
        if not self.props_area:
            return
        off = self.tab_scroll.get(self.active_tab, 0)
        max_off = max(0, self._props_count * 24 - self.props_area.h)
        self.tab_scroll[self.active_tab] = max(
            0, min(max_off, off - dy_steps * 48))

    def run_validation(self) -> ValidationReport:
        # Sensor availability comes from the agent's declarative suite —
        # the runtime SensorManager reference can go stale after a rebuild.
        self.last_validation_report = EnvironmentValidator.validate(
            road_def=self.road_def,
            agent=self.agent,
            entities=self.entities,
        )
        return self.last_validation_report

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

    def _get_sensor_config(self, name: Optional[str]):
        for cfg in getattr(self.agent, "sensor_configs", []):
            if cfg.name == name:
                return cfg
        return None

    def get_properties_for_active_tab(self) -> List[PropertyRow]:
        props: List[PropertyRow] = []

        # 1. OVERVIEW TAB
        if self.active_tab == "OVERVIEW":
            props.append(PropertyRow("ov_head_env", "--- ENVIRONMENT SUMMARY ---", "label", ""))
            props.append(PropertyRow("ov_name", "Name", "label", self.road_def.name))
            props.append(PropertyRow("ov_agent", "Agent", "label", f"{self.agent.agent_id} ({self.agent.entity_type})"))
            
            vec_dim = self.agent.observation_space.compute_vector_dim()
            props.append(PropertyRow("ov_obs", "Observations", "label", f"{vec_dim} features ({len(self.agent.observation_space.get_active_channels())} ch)"))
            
            act_dim = len(self.agent.action_space.channels)
            props.append(PropertyRow("ov_act", "Action Space", "label", f"{self.agent.action_space.space_type.upper()} ({act_dim} ch)"))
            
            active_rewards = len([c for c in self.agent.reward_function.components if c.enabled])
            props.append(PropertyRow("ov_rew", "Reward Function", "label", f"{active_rewards} components active"))
            
            active_terms = len([r for r in self.agent.termination_rules.rules if r.enabled])
            props.append(PropertyRow("ov_term", "Termination Rules", "label", f"{active_terms} conditions"))
            
            props.append(PropertyRow("ov_scen", "Active Scenario", "label", self.scenario_def.name))
            
            is_valid = self.last_validation_report.is_valid_for_rl if self.last_validation_report else True
            status_txt = "VALID (READY FOR RL)" if is_valid else f"INVALID ({len(self.last_validation_report.errors)} ERRORS)"
            props.append(PropertyRow("ov_val", "Training Readiness", "label", status_txt))

            props.append(PropertyRow("ov_head_actions", "--- TRAINING ACTIONS ---", "label", ""))
            props.append(PropertyRow("ov_btn_export", "Export Bundle", "action", "Export Training"))
            props.append(PropertyRow("ov_btn_tmpl_basic", "Template: Basic", "action", "Load Basic"))
            props.append(PropertyRow("ov_btn_tmpl_lane", "Template: Lane Keep", "action", "Load Lane"))
            props.append(PropertyRow("ov_btn_tmpl_obs", "Template: Obstacles", "action", "Load Obstacles"))

        # 2. AGENT DEFINITION TAB
        elif self.active_tab == "AGENT":
            a = self.agent
            props.append(PropertyRow("ag_id", "Agent ID", "label", a.agent_id))
            props.append(PropertyRow("ag_name", "Agent Name", "label", a.name))
            props.append(PropertyRow("ag_type", "Controlled Entity", "label", a.entity_type.upper()))
            props.append(PropertyRow("ag_ent_id", "Target Entity ID", "label", str(a.entity_id)))
            props.append(PropertyRow("ag_sensors", "Sensors Bound", "label", f"{len(a.sensor_names)} sensors"))
            props.append(PropertyRow("ag_spawn_speed", "Initial Spawn Speed", "float", a.spawn_config.initial_speed, 0.0, 60.0, 2.0, unit="m/s"))
            props.append(PropertyRow("ag_spawn_jit", "Spawn Lat Jitter", "float", a.spawn_config.lateral_jitter_m, 0.0, 5.0, 0.2, unit="m"))

        # 2b. SENSORS TAB — attached sensor suite (cameras, LiDAR, IMU)
        elif self.active_tab == "SENSORS":
            cfgs = getattr(self.agent, "sensor_configs", [])
            props.append(PropertyRow("sen_head", "--- SENSORS ON VEHICLE ---", "label", ""))
            for cfg in cfgs:
                props.append(PropertyRow(
                    f"sen_en_{cfg.name}",
                    f"{cfg.name} ({cfg.display_type})", "bool", cfg.enabled))
                props.append(PropertyRow(
                    f"sen_sel_{cfg.name}", "  Edit", "action",
                    "SELECT" if cfg.name != self.selected_sensor_name else "EDITING"))
            props.append(PropertyRow("sen_add_head", "--- ADD SENSOR ---", "label", ""))
            props.append(PropertyRow("sen_add_camera", "+ RGB Camera", "action", "ADD"))
            props.append(PropertyRow("sen_add_lidar", "+ LiDAR", "action", "ADD"))
            props.append(PropertyRow("sen_add_imu", "+ IMU", "action", "ADD"))
            props.append(PropertyRow("sen_add_state", "+ Vehicle State", "action", "ADD"))

            sel = self._get_sensor_config(self.selected_sensor_name)
            if sel is not None:
                p = sel.merged_params()
                props.append(PropertyRow("sen_hdr_sel",
                                         f"--- {sel.name.upper()} ---", "label",
                                         sel.display_type))
                props.append(PropertyRow("sen_rate", "Update Rate", "float",
                                         p.get("update_frequency_hz", 30.0),
                                         1.0, 120.0, 5.0, unit="Hz"))
                if sel.sensor_type == "camera_rgb":
                    props.append(PropertyRow("sen_fov", "FOV", "float",
                                             p["fov_degrees"], 30.0, 120.0, 5.0, unit="°"))
                    props.append(PropertyRow("sen_w", "Width", "int",
                                             p["width"], 32, 512, 16, unit="px"))
                    props.append(PropertyRow("sen_h", "Height", "int",
                                             p["height"], 32, 512, 16, unit="px"))
                    props.append(PropertyRow("sen_posx", "Mount X", "float",
                                             p["local_pos"][0], -5.0, 5.0, 0.1, unit="m"))
                    props.append(PropertyRow("sen_posy", "Mount Y", "float",
                                             p["local_pos"][1], -5.0, 5.0, 0.1, unit="m"))
                    props.append(PropertyRow("sen_posz", "Mount Z", "float",
                                             p["local_pos"][2], -5.0, 5.0, 0.1, unit="m"))
                    props.append(PropertyRow("sen_yaw", "Mount Yaw", "float",
                                             p["local_yaw"], -180.0, 180.0, 5.0, unit="°"))
                    props.append(PropertyRow("sen_pitch", "Mount Pitch", "float",
                                             math.degrees(p["local_pitch"]),
                                             -90.0, 90.0, 5.0, unit="°"))
                    props.append(PropertyRow("sen_roll", "Mount Roll", "float",
                                             math.degrees(p.get("local_roll", 0.0)),
                                             -90.0, 90.0, 5.0, unit="°"))
                    props.append(PropertyRow("sen_near", "Near Clip", "float",
                                             p.get("near_clip", 0.5),
                                             0.05, 10.0, 0.05, unit="m"))
                    props.append(PropertyRow("sen_far", "Far Clip", "float",
                                             p.get("far_clip", 1000.0),
                                             50.0, 5000.0, 50.0, unit="m"))
                    props.append(PropertyRow("sen_noise", "Pixel Noise", "float",
                                             p["noise_std"], 0.0, 0.5, 0.02))
                    props.append(PropertyRow("sen_latency", "Latency", "float",
                                             p["latency_seconds"], 0.0, 1.0, 0.02, unit="s"))
                    in_obs = any(s["name"] == sel.name
                                 for s in self.agent.observation_space.image_channel_specs())
                    props.append(PropertyRow("sen_in_obs", "In Observations", "bool", in_obs))
                elif sel.sensor_type == "lidar_rays":
                    props.append(PropertyRow("sen_beams", "Beams", "int",
                                             p["num_rays"], 3, 64, 1))
                    props.append(PropertyRow("sen_fov", "FOV", "float",
                                             p["fov_degrees"], 30.0, 360.0, 10.0, unit="°"))
                    props.append(PropertyRow("sen_range", "Range", "float",
                                             p["max_range"], 5.0, 200.0, 5.0, unit="m"))
                    props.append(PropertyRow("sen_posx", "Mount X", "float",
                                             p["local_pos"][0], -5.0, 5.0, 0.1, unit="m"))
                    props.append(PropertyRow("sen_posy", "Mount Y", "float",
                                             p["local_pos"][1], -5.0, 5.0, 0.1, unit="m"))
                    props.append(PropertyRow("sen_posz", "Mount Z", "float",
                                             p["local_pos"][2], -5.0, 5.0, 0.1, unit="m"))
                    props.append(PropertyRow("sen_yaw", "Mount Yaw", "float",
                                             p["local_yaw"], -180.0, 180.0, 5.0, unit="°"))
                    props.append(PropertyRow("sen_noise", "Range Noise", "float",
                                             p["noise_std"], 0.0, 0.5, 0.01, unit="m"))
                elif sel.sensor_type == "imu":
                    props.append(PropertyRow("sen_anoise", "Accel Noise", "float",
                                             p["accel_noise_std"], 0.0, 1.0, 0.01))
                    props.append(PropertyRow("sen_gnoise", "Gyro Noise", "float",
                                             p["gyro_noise_std"], 0.0, 1.0, 0.005))
                    props.append(PropertyRow("sen_drift", "Bias Drift", "float",
                                             p["bias_drift_rate"], 0.0, 0.1, 0.001))
                else:
                    props.append(PropertyRow("sen_noise", "Noise", "float",
                                             p.get("noise_std", 0.0), 0.0, 1.0, 0.01))
                    props.append(PropertyRow("sen_latency", "Latency", "float",
                                             p.get("latency_seconds", 0.0), 0.0, 1.0, 0.02, unit="s"))
                props.append(PropertyRow("sen_dup", "Duplicate Sensor", "action", "DUPLICATE"))
                props.append(PropertyRow("sen_remove", "Remove Sensor", "action", "REMOVE"))
            else:
                props.append(PropertyRow("sen_none", "No Sensor Selected", "label",
                                         "Click EDIT beside a sensor"))

        # 3. OBSERVATION SPACE DESIGNER TAB
        elif self.active_tab == "OBS":
            obs = self.agent.observation_space
            props.append(PropertyRow("obs_dim", "Vector Dim", "label", f"{obs.compute_vector_dim()} features"))
            props.append(PropertyRow("obs_flatten", "Flatten to 1D Box", "bool", obs.flatten_vector))
            is_leak_free, _ = obs.validate_no_leakage()
            props.append(PropertyRow("obs_leak_free", "Leakage Guard", "label", "SECURE (NO ORACLE)" if is_leak_free else "LEAK DETECTED"))
            
            # Channel toggles
            for ch in obs.channels:
                props.append(PropertyRow(f"obs_ch_{ch.name}", ch.name, "bool", ch.enabled))

        # 4. ACTION SPACE DESIGNER TAB
        elif self.active_tab == "ACTION":
            act = self.agent.action_space
            props.append(PropertyRow("act_type", "Space Mode", "enum", act.space_type, options=["continuous", "discrete"]))
            for ch in act.channels:
                props.append(PropertyRow(f"act_head_{ch.name}", f"--- {ch.name.upper()} CHANNEL ---", "label", ""))
                props.append(PropertyRow(f"act_min_{ch.name}", "Min Limit", "float", ch.min_val, -1.0, 0.0, 0.1))
                props.append(PropertyRow(f"act_max_{ch.name}", "Max Limit", "float", ch.max_val, 0.0, 1.0, 0.1))
                props.append(PropertyRow(f"act_dz_{ch.name}", "Dead Zone", "float", ch.dead_zone, 0.0, 0.2, 0.01))
                props.append(PropertyRow(f"act_rl_{ch.name}", "Rate Limit", "float", ch.rate_limit, 0.0, 20.0, 0.5, unit="/s"))

        # 5. REWARD FUNCTION DESIGNER TAB
        elif self.active_tab == "REWARD":
            rf = self.agent.reward_function
            for comp in rf.components:
                props.append(PropertyRow(f"rf_en_{comp.component_id}", f"[{'ON' if comp.enabled else 'OFF'}] {comp.name}", "bool", comp.enabled))
                props.append(PropertyRow(f"rf_wt_{comp.component_id}", f"  Weight", "float", comp.weight, -200.0, 500.0, 1.0 if abs(comp.weight) >= 1.0 else 0.05))

        # 6. TERMINATION DESIGNER TAB
        elif self.active_tab == "TERM":
            td = self.agent.termination_rules
            for rule in td.rules:
                tag = "TRUNC" if rule.is_truncation else "TERM"
                props.append(PropertyRow(f"term_en_{rule.rule_id}", f"[{tag}] {rule.name}", "bool", rule.enabled))

        # 7. SCENARIO & RANDOMIZATION TAB
        elif self.active_tab == "SCENARIO":
            sc = self.scenario_def
            props.append(PropertyRow("scen_select", "Scenario Preset", "enum", sc.name, options=[
                "Basic Lane Following", "High Speed Racing", "Wet Track Adverse Weather",
                "Obstacle Evasion", "Sensor Noise Challenge", "Full Domain Randomization"
            ]))
            props.append(PropertyRow("scen_weather", "Weather", "enum", sc.weather, options=["clear", "rain", "fog"]))
            props.append(PropertyRow("scen_time", "Time of Day", "enum", sc.time_of_day, options=["day", "dusk", "night"]))
            props.append(PropertyRow("scen_friction", "Friction Mult", "float", sc.surface_friction_mult, 0.2, 2.0, 0.05))
            props.append(PropertyRow("scen_light", "Ambient Light", "float", sc.ambient_light, 0.1, 1.0, 0.05))
            props.append(PropertyRow("scen_rand_en", "Domain Randomization", "bool", sc.randomization.enabled))

        # 8. VALIDATION TAB
        elif self.active_tab == "VALIDATE":
            rep = self.last_validation_report
            if rep:
                props.append(PropertyRow("val_gate", "Training Readiness", "label", "PASS - READY" if rep.is_valid_for_rl else "FAIL - BLOCKED"))
                props.append(PropertyRow("val_errs", f"Errors ({len(rep.errors)})", "label", "None" if not rep.errors else rep.errors[0].message[:28]))
                props.append(PropertyRow("val_warns", f"Warnings ({len(rep.warnings)})", "label", f"{len(rep.warnings)} items"))
                props.append(PropertyRow("val_infos", f"Info ({len(rep.infos)})", "label", f"{len(rep.infos)} items"))
                props.append(PropertyRow("val_btn_run", "Re-Run Validation", "action", "Validate Now"))
                for i, err in enumerate(rep.errors[:8]):
                    props.append(PropertyRow(f"val_nav_err_{i}", f"ERROR: {err.subsystem}", "nav", err.message[:32]))
                for i, w in enumerate(rep.warnings[:6]):
                    props.append(PropertyRow(f"val_nav_warn_{i}", f"WARN: {w.subsystem}", "nav", w.message[:32]))

        # 8b. TRAINING & EXPERIMENTS TAB
        elif self.active_tab == "TRAIN":
            data = self.train_provider() if self.train_provider else {}
            exps = data.get("experiments", [])
            sel_exp = data.get("selected_experiment")
            sel_run = data.get("selected_run") or {}

            props.append(PropertyRow("trn_head_exp", "--- EXPERIMENTS ---", "label", ""))
            props.append(PropertyRow("trn_create", "Create Experiment", "action", "CREATE FROM ENV"))
            if not exps:
                props.append(PropertyRow("trn_none", "No experiments", "label", "Create one first"))
            for i, e in enumerate(exps[:6]):
                tag = "*" if e.get("experiment_id") == sel_exp else " "
                label = f"{tag} {e.get('name','')[:18]} [{e.get('algorithm','')}]"
                status = "LAUNCHED" if e.get("launched") else "draft"
                props.append(PropertyRow(f"trn_sel_{i}", label, "action", status.upper()))

            if sel_exp:
                props.append(PropertyRow("trn_head_run", "--- RUN ---", "label", ""))
                algo = (data.get("selected_algorithm") or "ppo").upper()
                props.append(PropertyRow("trn_launch", "Launch Training", "action",
                                         f"LAUNCH {algo}"))
                props.append(PropertyRow("trn_batch", "Batch x2 Seeds", "action", "RUN BATCH"))
                run_status = sel_run.get("status", "")
                if run_status in ("RUNNING", "PAUSED", "STARTING", "QUEUED"):
                    props.append(PropertyRow("trn_cancel", "Cancel Run", "action", "CANCEL"))
                if sel_run.get("checkpoints"):
                    props.append(PropertyRow("trn_resume", "Resume From Checkpoint", "action", "RESUME"))
                    props.append(PropertyRow("trn_eval", "Evaluate Checkpoint", "action", "EVALUATE"))
                props.append(PropertyRow("trn_repro", "Reproducibility Check", "action", "VERIFY"))
                props.append(PropertyRow("trn_dataset", "Export Dataset", "action", "EXPORT DATASET"))
                props.append(PropertyRow("trn_export", "Export Experiment", "action", "EXPORT"))

                if sel_run:
                    props.append(PropertyRow("trn_head_mon", "--- RUN MONITOR ---", "label", ""))
                    props.append(PropertyRow("trn_status", "Status", "label", run_status or "no runs"))
                    props.append(PropertyRow("trn_steps", "Timesteps", "label", str(sel_run.get("current_timestep", 0))))
                    props.append(PropertyRow("trn_eps", "Episodes", "label", str(sel_run.get("episode_count", 0))))
                    curr = sel_run.get("curriculum")
                    if curr:
                        props.append(PropertyRow("trn_curr", "Curriculum Stage",
                                                 "label",
                                                 f"{curr.get('stage_index')}: {curr.get('stage_name')}"))
                        props.append(PropertyRow("trn_curr_eps", "Stage Episodes",
                                                 "label",
                                                 str(curr.get("episodes_in_stage", 0))))
                    for k, v in list(sel_run.get("latest_metrics", {}).items())[:6]:
                        props.append(PropertyRow(f"trn_m_{k}", k, "label",
                                                 f"{v:.3f}" if isinstance(v, (int, float)) else str(v)[:16]))
                    series = sel_run.get("reward_series") or []
                    if series:
                        props.append(PropertyRow("trn_chart_rew", "Reward / Episode", "chart", series))
                    if sel_run.get("error"):
                        props.append(PropertyRow("trn_err", "Error", "label",
                                                 str(sel_run["error"].get("message", ""))[:28]))

            batch = data.get("batch_status")
            if batch:
                props.append(PropertyRow("trn_head_batch", "--- BATCH ---", "label", ""))
                props.append(PropertyRow("trn_batch_id", "Batch", "label",
                                         batch.get("batch_id", "")[:24]))
                props.append(PropertyRow("trn_batch_prog", "Progress", "label",
                                         f"{batch.get('finished',0)}/{batch.get('total',0)} "
                                         f"run={batch.get('running',0)} "
                                         f"ok={batch.get('completed',0)} "
                                         f"fail={batch.get('failed',0)}"))
                props.append(PropertyRow("trn_batch_cancel", "Cancel Batch", "action", "CANCEL BATCH"))

            props.append(PropertyRow("trn_head_workers", "--- WORKERS ---", "label", ""))
            props.append(PropertyRow("trn_w_serve", "Start local worker",
                                     "action", "SERVE WORKER"))
            workers = data.get("workers") or []
            if workers:
                for i, wrow in enumerate(workers[:8]):
                    tag = {"RUNNING": "*", "OFFLINE": "!", "IDLE": " "}.get(
                        wrow.get("status"), "?")
                    job = wrow.get("job") or "-"
                    hb = wrow.get("heartbeat_age_s")
                    hb_txt = f" hb={hb}s" if hb is not None else ""
                    props.append(PropertyRow(
                        f"trn_w_{i}", f"{tag} {str(wrow.get('worker_id'))[:18]}",
                        "label",
                        f"{wrow.get('status','?')} job={job}{hb_txt}"))
                    if wrow.get("status") != "OFFLINE":
                        props.append(PropertyRow(
                            f"trn_w_off_{i}",
                            f"  Mark offline ({str(wrow.get('worker_id'))[:16]})",
                            "action", "MARK OFFLINE"))

            ds_prev = data.get("dataset_preview")
            if ds_prev:
                props.append(PropertyRow("trn_head_ds", "--- DATASET ---", "label", ""))
                props.append(PropertyRow("trn_ds_valid", "Validation", "label",
                                         "VALID" if ds_prev.get("valid") else "INVALID"))
                stats = ds_prev.get("stats") or {}
                props.append(PropertyRow("trn_ds_eps", "Episodes", "label",
                                         str(stats.get("episode_count", 0))))
                props.append(PropertyRow("trn_ds_steps", "Steps", "label",
                                         str(stats.get("step_count", 0))))
                ret = stats.get("return") or {}
                props.append(PropertyRow("trn_ds_ret", "Return mean±std", "label",
                                         f"{ret.get('mean', 0):.2f}±{ret.get('std', 0):.2f}"))
                reasons = stats.get("termination_reasons") or {}
                for i, (rsn, cnt) in enumerate(
                        sorted(reasons.items(), key=lambda kv: -kv[1])[:3]):
                    props.append(PropertyRow(f"trn_ds_r_{i}", rsn[:20], "label", str(cnt)))
                for i, ep in enumerate(ds_prev.get("episodes", [])[:6]):
                    props.append(PropertyRow(
                        f"trn_ds_e_{i}",
                        f"  {ep.get('episode_id','?')[:16]}", "label",
                        f"r={ep.get('total_return',0):.1f} "
                        f"n={ep.get('length',0)}"))

            cmp_chart = data.get("comparison_chart") or []
            if cmp_chart:
                props.append(PropertyRow("trn_head_cmp", "--- COMPARISON (reward) ---", "label", ""))
                props.append(PropertyRow("trn_cmp_chart", "Smoothed reward",
                                         "multichart", cmp_chart))
                for i, s in enumerate(cmp_chart[:4]):
                    props.append(PropertyRow(f"trn_cmp_{i}", s["label"][:20], "label",
                                             f"last={s['data'][-1]:.1f}" if s["data"] else "-"))
            else:
                cmp_data = data.get("comparison")
                if cmp_data and cmp_data.get("series"):
                    props.append(PropertyRow("trn_head_cmp", "--- COMPARISON (reward) ---", "label", ""))
                    for i, s in enumerate(cmp_data["series"][:4]):
                        props.append(PropertyRow(f"trn_cmp_{i}", s["label"][:20], "label",
                                                 f"mean={s['mean']:.1f} best={s['max']:.1f}"))
            if sel_exp:
                props.append(PropertyRow("trn_compare", "Compare Runs", "action", "COMPARE"))

        # 9. SCENE HIERARCHY
        elif self.active_tab == "SCENE":
            props.append(PropertyRow("sc_head_geom", "--- TRACK GEOMETRY ---", "label", ""))
            props.append(PropertyRow("sc_track_info", f"Track: {self.road_def.name}", "label", f"{len(self.road_def.control_points)} pts"))
            props.append(PropertyRow("sc_spawn_info", "Spawn Point", "label", f"({self.road_def.spawn_point.x:.0f}, {self.road_def.spawn_point.y:.0f})"))
            props.append(PropertyRow("sc_cp_info", "Checkpoints", "label", f"{self.road_def.num_checkpoints} gates"))

            props.append(PropertyRow("sc_head_ent", f"--- ENTITIES ({len(self.entities)}) ---", "label", ""))
            for i, ent in enumerate(self.entities):
                props.append(PropertyRow(f"sc_ent_sel_{ent.entity_id}", ent.name, "action", f"SELECT #{ent.entity_id}"))

            props.append(PropertyRow("sc_head_add", "--- PLACE NEW ENTITY ---", "label", ""))
            props.append(PropertyRow("sc_add_obstacle", "+ Box Obstacle", "action", "PLACE BOX"))
            props.append(PropertyRow("sc_add_barrier", "+ Concrete Barrier", "action", "PLACE BARRIER"))
            props.append(PropertyRow("sc_add_cone", "+ Traffic Cone", "action", "PLACE CONE"))
            props.append(PropertyRow("sc_add_sign", "+ Traffic Sign", "action", "PLACE SIGN"))
            props.append(PropertyRow("sc_add_light", "+ Traffic Light", "action", "PLACE LIGHT"))

        # 10. TRACK DEFINITION
        elif self.active_tab == "TRACK":
            r = self.road_def
            b = r.boundary_config
            props.append(PropertyRow("track_closed", "Closed Circuit (laps)", "bool", r.is_closed))
            props.append(PropertyRow("track_friction", "Surface Friction", "float", r.default_friction, 0.1, 2.5, 0.05))
            props.append(PropertyRow("track_checkpoints", "Checkpoints", "int", r.num_checkpoints, 4, 64, 2))
            props.append(PropertyRow("b_left_type", "Left Boundary", "enum", b.left_type, options=["guardrail", "wall", "curb", "open"]))
            props.append(PropertyRow("b_right_type", "Right Boundary", "enum", b.right_type, options=["guardrail", "wall", "curb", "open"]))
            props.append(PropertyRow("b_has_curbs", "Curb Ribbons", "bool", b.has_curbs))
            props.append(PropertyRow("b_curb_width", "Curb Width", "float", b.curb_width, 0.2, 2.0, 0.1, unit="m"))
            props.append(PropertyRow("b_wall_height", "Wall Height", "float", b.wall_height, 0.3, 3.0, 0.1, unit="m"))

            # Spawn point — sp_* handlers below mutate road_def.spawn_point
            sp = r.spawn_point
            props.append(PropertyRow("sp_head", "--- SPAWN POINT ---", "label", ""))
            props.append(PropertyRow("sp_pos_x", "Spawn X", "float", sp.x, -1000.0, 1000.0, 1.0, unit="m"))
            props.append(PropertyRow("sp_pos_y", "Spawn Y", "float", sp.y, -1000.0, 1000.0, 1.0, unit="m"))
            props.append(PropertyRow("sp_elevation", "Spawn Elevation", "float", sp.z, -50.0, 100.0, 0.5, unit="m"))
            props.append(PropertyRow("sp_yaw", "Spawn Heading", "float", math.degrees(sp.yaw), -180.0, 180.0, 5.0, unit="°"))
            props.append(PropertyRow("sp_speed", "Initial Speed", "float", sp.initial_speed, 0.0, 60.0, 1.0, unit="m/s"))

        # 11. CONTROL POINT
        elif self.active_tab == "POINT":
            if self.selected_point_idx is not None and 0 <= self.selected_point_idx < len(self.road_def.control_points):
                cp = self.road_def.control_points[self.selected_point_idx]
                props.append(PropertyRow("cp_idx_display", f"Point #{self.selected_point_idx}", "label", f"Total: {len(self.road_def.control_points)}"))
                props.append(PropertyRow("cp_pos_x", "Coord X", "float", cp.x, -1000.0, 1000.0, 1.0, unit="m"))
                props.append(PropertyRow("cp_pos_y", "Coord Y", "float", cp.y, -1000.0, 1000.0, 1.0, unit="m"))
                props.append(PropertyRow("cp_width", "Road Width", "float", cp.width, 4.0, 40.0, 1.0, unit="m"))
                props.append(PropertyRow("cp_elevation", "Elevation (Z)", "float", cp.z, -50.0, 100.0, 0.5, unit="m"))
                props.append(PropertyRow("cp_banking", "Banking", "float", cp.banking, -30.0, 30.0, 1.0, unit="°"))
                props.append(PropertyRow("cp_action_del", "Delete Point", "action", "Delete Point"))
            else:
                props.append(PropertyRow("cp_none", "No Point Selected", "label", "Click point on canvas"))

        # 12. PLACED WORLD ENTITY
        elif self.active_tab == "ENTITY":
            ent = self.get_selected_entity()
            if ent:
                props.append(PropertyRow("ent_name", f"Entity: {ent.name}", "label", f"ID: {ent.entity_id}"))
                props.append(PropertyRow("ent_pos_x", "Position X", "float", ent.pos.x, -1000.0, 1000.0, 1.0, unit="m"))
                props.append(PropertyRow("ent_pos_y", "Position Y", "float", ent.pos.y, -1000.0, 1000.0, 1.0, unit="m"))
                props.append(PropertyRow("ent_yaw", "Heading (Yaw)", "float", math.degrees(ent.yaw), -180.0, 180.0, 5.0, unit="°"))
                if hasattr(ent, 'is_collidable'):
                    props.append(PropertyRow("ent_col", "Collidable", "bool", ent.is_collidable))
                props.append(PropertyRow("ent_del", "Delete Entity", "action", "Delete Entity"))
            else:
                props.append(PropertyRow("ent_none", "No Entity Selected", "label", "Click entity on canvas"))

        return props

    def handle_nav(self, prop_id: str) -> None:
        """Navigate to the owning tab for a validation issue row."""
        rep = self.last_validation_report
        issue = None
        if prop_id.startswith("val_nav_err_") and rep:
            idx = int(prop_id.rsplit("_", 1)[-1])
            if idx < len(rep.errors):
                issue = rep.errors[idx]
        elif prop_id.startswith("val_nav_warn_") and rep:
            idx = int(prop_id.rsplit("_", 1)[-1])
            if idx < len(rep.warnings):
                issue = rep.warnings[idx]
        if issue is not None:
            self.active_tab = self.SUBSYSTEM_TAB.get(
                issue.subsystem.lower(), "VALIDATE")

    def _sync_image_channels(self) -> None:
        """Keep obs-space image channels in sync with enabled cameras."""
        obs = self.agent.observation_space
        cams = [c for c in getattr(self.agent, "sensor_configs", [])
                if c.enabled and c.sensor_type == "camera_rgb"]
        obs.image_channels = [
            {"name": c.name,
             "shape": [int(c.merged_params()["height"]),
                       int(c.merged_params()["width"]), 3]}
            for c in cams]

    def _unique_sensor_name(self, base: str) -> str:
        existing = {c.name for c in getattr(self.agent, "sensor_configs", [])}
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def _handle_sensor_prop(self, prop_id: str, delta_or_value: Any) -> None:
        """Mutations for the SENSORS tab — suite edits + per-sensor params."""
        from sim_env.sensor_config import SensorConfig
        cfgs = getattr(self.agent, "sensor_configs", [])
        if cfgs is None:
            self.agent.sensor_configs = cfgs = []

        if prop_id.startswith("sen_en_"):
            name = prop_id[len("sen_en_"):]
            cfg = self._get_sensor_config(name)
            if cfg is not None:
                cfg.enabled = not cfg.enabled
                self._sync_image_channels()
        elif prop_id.startswith("sen_sel_"):
            name = prop_id[len("sen_sel_"):]
            self.selected_sensor_name = name
        elif prop_id.startswith("sen_add_"):
            stype = {"sen_add_camera": "camera_rgb",
                     "sen_add_lidar": "lidar_rays",
                     "sen_add_imu": "imu",
                     "sen_add_state": "vehicle_state"}[prop_id]
            base = {"camera_rgb": "camera", "lidar_rays": "lidar",
                    "imu": "imu", "vehicle_state": "state"}[stype]
            name = self._unique_sensor_name(
                "rgb_camera" if stype == "camera_rgb" and
                "rgb_camera" not in {c.name for c in cfgs} else base)
            cfg = SensorConfig.for_type(stype, name)
            cfgs.append(cfg)
            self.selected_sensor_name = name
            self._sync_image_channels()
        else:
            cfg = self._get_sensor_config(self.selected_sensor_name)
            if cfg is None:
                return
            p = cfg.params
            d = delta_or_value or 0.0
            if prop_id == "sen_remove":
                cfgs[:] = [c for c in cfgs if c.name != cfg.name]
                self.selected_sensor_name = cfgs[0].name if cfgs else ""
                self._sync_image_channels()
            elif prop_id == "sen_dup":
                dup = SensorConfig.from_dict(cfg.to_dict())
                dup.name = self._unique_sensor_name(cfg.name)
                cfgs.insert(cfgs.index(cfg) + 1, dup)
                self.selected_sensor_name = dup.name
                self._sync_image_channels()
            elif prop_id == "sen_rate":
                p["update_frequency_hz"] = float(
                    max(1.0, min(120.0, p.get("update_frequency_hz", 30.0) + d)))
            elif prop_id == "sen_fov":
                p["fov_degrees"] = float(
                    max(10.0, min(360.0, cfg.merged_params()["fov_degrees"] + d)))
            elif prop_id == "sen_w":
                p["width"] = int(max(32, min(512,
                                 cfg.merged_params()["width"] + d)))
                self._sync_image_channels()
            elif prop_id == "sen_h":
                p["height"] = int(max(32, min(512,
                                  cfg.merged_params()["height"] + d)))
                self._sync_image_channels()
            elif prop_id in ("sen_posx", "sen_posy", "sen_posz"):
                lp = list(p.get("local_pos") or cfg.merged_params()["local_pos"])
                idx = {"sen_posx": 0, "sen_posy": 1, "sen_posz": 2}[prop_id]
                lp[idx] = round(lp[idx] + d, 2)
                p["local_pos"] = lp
            elif prop_id == "sen_yaw":
                deg = (cfg.merged_params()["local_yaw"] + d) % 360.0
                if deg > 180.0:
                    deg -= 360.0
                p["local_yaw"] = deg
            elif prop_id == "sen_pitch":
                # params store radians; the row displays degrees
                deg = math.degrees(cfg.merged_params()["local_pitch"]) + d
                p["local_pitch"] = math.radians(max(-90.0, min(90.0, deg)))
            elif prop_id == "sen_roll":
                deg = math.degrees(cfg.merged_params().get("local_roll", 0.0)) + d
                p["local_roll"] = math.radians(max(-90.0, min(90.0, deg)))
            elif prop_id == "sen_near":
                p["near_clip"] = round(max(0.05, min(10.0,
                    cfg.merged_params().get("near_clip", 0.5) + d)), 3)
            elif prop_id == "sen_far":
                p["far_clip"] = round(max(50.0, min(5000.0,
                    cfg.merged_params().get("far_clip", 1000.0) + d)), 1)
            elif prop_id == "sen_noise":
                key = "noise_std"
                p[key] = round(max(0.0, min(1.0,
                               cfg.merged_params().get(key, 0.0) + d)), 3)
            elif prop_id == "sen_latency":
                p["latency_seconds"] = round(max(0.0, min(1.0,
                    cfg.merged_params().get("latency_seconds", 0.0) + d)), 3)
            elif prop_id == "sen_beams":
                p["num_rays"] = int(max(3, min(64,
                                    cfg.merged_params()["num_rays"] + d)))
            elif prop_id == "sen_range":
                p["max_range"] = float(max(5.0, min(200.0,
                    cfg.merged_params()["max_range"] + d)))
            elif prop_id == "sen_anoise":
                p["accel_noise_std"] = round(max(0.0, min(1.0,
                    cfg.merged_params()["accel_noise_std"] + d)), 3)
            elif prop_id == "sen_gnoise":
                p["gyro_noise_std"] = round(max(0.0, min(1.0,
                    cfg.merged_params()["gyro_noise_std"] + d)), 4)
            elif prop_id == "sen_drift":
                p["bias_drift_rate"] = round(max(0.0, min(0.1,
                    cfg.merged_params()["bias_drift_rate"] + d)), 4)
            elif prop_id == "sen_in_obs":
                obs = self.agent.observation_space
                specs = obs.image_channel_specs()
                if any(s["name"] == cfg.name for s in specs):
                    specs = [s for s in specs if s["name"] != cfg.name]
                else:
                    m = cfg.merged_params()
                    specs.append({"name": cfg.name,
                                  "shape": [int(m["height"]),
                                            int(m["width"]), 3]})
                obs.image_channels = specs

        # Derived name list stays in sync for legacy consumers/validation
        self.agent.sensor_names = [c.name for c in cfgs if c.enabled]

    def handle_property_change(self, prop_id: str, delta_or_value: Any) -> None:
        """Applies property updates directly to underlying data models with validation."""
        r = self.road_def
        b = r.boundary_config
        act = self.agent.action_space
        obs = self.agent.observation_space
        rf = self.agent.reward_function
        td = self.agent.termination_rules
        sc = self.scenario_def

        # OVERVIEW Action Buttons
        if prop_id == "ov_btn_export":
            if self.on_export_training:
                self.on_export_training()
        elif prop_id == "ov_btn_tmpl_basic":
            if self.on_load_template:
                self.on_load_template("basic_driving")
        elif prop_id == "ov_btn_tmpl_lane":
            if self.on_load_template:
                self.on_load_template("lane_following")
        elif prop_id == "ov_btn_tmpl_obs":
            if self.on_load_template:
                self.on_load_template("obstacle_avoidance")

        # VALIDATE Action
        elif prop_id == "val_btn_run":
            self.run_validation()

        # Track
        elif prop_id == "track_closed":
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

        # Spawn Point
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

        # Vehicle
        elif prop_id.startswith("vc_"):
            vc = self.vehicle_config
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

        # Placed world entity
        elif prop_id.startswith("ent_"):
            ent = self.get_selected_entity()
            if prop_id == "ent_del":
                if ent is not None and ent in self.entities:
                    self.entities.remove(ent)
                    self.selected_entity_id = None
            elif ent is not None:
                if prop_id == "ent_pos_x":
                    ent.pos.x = round(ent.pos.x + delta_or_value, 1)
                elif prop_id == "ent_pos_y":
                    ent.pos.y = round(ent.pos.y + delta_or_value, 1)
                elif prop_id == "ent_yaw":
                    deg = (math.degrees(ent.yaw) + delta_or_value) % 360.0
                    if deg > 180.0:
                        deg -= 360.0
                    ent.yaw = math.radians(deg)
                elif prop_id == "ent_col" and hasattr(ent, 'is_collidable'):
                    ent.is_collidable = not ent.is_collidable

        # Sensor suite (SENSORS tab)
        elif prop_id.startswith("sen_"):
            self._handle_sensor_prop(prop_id, delta_or_value)

        # Control Point
        elif prop_id.startswith("cp_") and self.selected_point_idx is not None:
            if 0 <= self.selected_point_idx < len(r.control_points):
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
                elif prop_id == "cp_action_del":
                    if len(r.control_points) > 3:
                        r.control_points.pop(self.selected_point_idx)
                        self.selected_point_idx = max(0, self.selected_point_idx - 1)

        # Observations
        elif prop_id == "obs_flatten":
            obs.flatten_vector = not obs.flatten_vector
        elif prop_id.startswith("obs_ch_"):
            ch_name = prop_id.replace("obs_ch_", "")
            for ch in obs.channels:
                if ch.name == ch_name:
                    ch.enabled = not ch.enabled

        # Actions
        elif prop_id == "act_type":
            act.space_type = "discrete" if act.space_type == "continuous" else "continuous"
        elif prop_id.startswith("act_dz_"):
            ch_name = prop_id.replace("act_dz_", "")
            for ch in act.channels:
                if ch.name == ch_name:
                    ch.dead_zone = round(max(0.0, min(0.3, ch.dead_zone + delta_or_value)), 3)
        elif prop_id.startswith("act_rl_"):
            ch_name = prop_id.replace("act_rl_", "")
            for ch in act.channels:
                if ch.name == ch_name:
                    ch.rate_limit = round(max(0.0, min(30.0, ch.rate_limit + delta_or_value)), 1)
        elif prop_id.startswith("act_min_"):
            ch_name = prop_id.replace("act_min_", "")
            for ch in act.channels:
                if ch.name == ch_name:
                    ch.min_val = round(max(-1.0, min(min(0.0, ch.max_val), ch.min_val + delta_or_value)), 2)
        elif prop_id.startswith("act_max_"):
            ch_name = prop_id.replace("act_max_", "")
            for ch in act.channels:
                if ch.name == ch_name:
                    ch.max_val = round(min(1.0, max(max(0.0, ch.min_val), ch.max_val + delta_or_value)), 2)

        # Rewards
        elif prop_id.startswith("rf_en_"):
            cid = prop_id.replace("rf_en_", "")
            comp = rf.get_component(cid)
            if comp:
                comp.enabled = not comp.enabled
        elif prop_id.startswith("rf_wt_"):
            cid = prop_id.replace("rf_wt_", "")
            comp = rf.get_component(cid)
            if comp:
                comp.weight = round(comp.weight + delta_or_value, 2)

        # Termination
        elif prop_id.startswith("term_en_"):
            rid = prop_id.replace("term_en_", "")
            rule = td.get_rule(rid)
            if rule:
                rule.enabled = not rule.enabled

        # Scenario
        elif prop_id == "scen_select":
            import copy
            from dataclasses import fields as _dc_fields
            presets = ScenarioDefinition.get_standard_scenarios()
            names = [s.name for s in presets.values()]
            try:
                idx = (names.index(sc.name) + 1) % len(names)
            except ValueError:
                idx = 0
            src = list(presets.values())[idx]
            # in-place copy keeps the shared ref: env + project pick it up
            for f in _dc_fields(src):
                setattr(sc, f.name, copy.deepcopy(getattr(src, f.name)))
        elif prop_id == "scen_weather":
            opts = ["clear", "rain", "fog"]
            try:
                sc.weather = opts[(opts.index(sc.weather) + 1) % len(opts)]
            except ValueError:
                sc.weather = opts[0]
        elif prop_id == "scen_time":
            opts = ["day", "dusk", "night"]
            try:
                sc.time_of_day = opts[(opts.index(sc.time_of_day) + 1) % len(opts)]
            except ValueError:
                sc.time_of_day = opts[0]
        elif prop_id == "scen_friction":
            sc.surface_friction_mult = round(max(0.2, min(2.0, sc.surface_friction_mult + delta_or_value)), 2)
        elif prop_id == "scen_light":
            sc.ambient_light = round(max(0.1, min(1.0, sc.ambient_light + delta_or_value)), 2)
        elif prop_id == "scen_rand_en":
            sc.randomization.enabled = not sc.randomization.enabled

        # Training & Experiments actions route to the application service layer
        elif prop_id.startswith("trn_"):
            if self.on_train_action:
                self.on_train_action(prop_id)

        # Auto-revalidate
        self.run_validation()

        # If agent spaces or rules changed, notify runtime to recompile pipelines
        if prop_id.startswith(("obs_", "act_", "rf_", "term_", "ag_", "sen_")) and self.on_agent_modified:
            self.on_agent_modified()

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
        Renders the complete 2-tier Environment Designer panel.
        """
        pygame.draw.rect(surface, T.C.panel, (x, y, w, h), border_radius=8)
        pygame.draw.rect(surface, T.C.border, (x, y, w, h), 1, border_radius=8)

        f_bold = fonts['bold']
        f_small = fonts['small']
        f_mono = fonts['mono']
        clickable_buttons: List[Tuple[pygame.Rect, str]] = []

        # 1. Header
        header_y = y + 8
        surface.blit(f_bold.render("ENVIRONMENT", True, T.C.accent_line), (x + 12, header_y))

        # 2. Two-tier category tab bar (abbreviated labels + issue badges)
        rep = self.last_validation_report
        error_tabs = set()
        if rep:
            for issue in rep.errors:
                error_tabs.add(self.SUBSYSTEM_TAB.get(
                    issue.subsystem.lower(), "VALIDATE"))
        tab_y1 = y + 32
        tab_w1 = (w - 20) // len(self.CATEGORY_GEO)
        for i, tab in enumerate(self.CATEGORY_GEO):
            tab_rect = pygame.Rect(x + 10 + i * tab_w1, tab_y1, tab_w1 - 2, 22)
            is_sel = (self.active_tab == tab)
            bg = T.C.accent if is_sel else T.C.panel_alt
            txt_col = T.C.text_on_accent if is_sel else T.C.text_dim
            pygame.draw.rect(surface, bg, tab_rect, border_radius=3)
            lbl = f_small.render(self.TAB_LABELS.get(tab, tab), True, txt_col)
            surface.blit(lbl, (tab_rect.centerx - lbl.get_width() // 2, tab_rect.centery - lbl.get_height() // 2))
            if tab in error_tabs:
                pygame.draw.circle(surface, T.C.error,
                                   (tab_rect.right - 6, tab_rect.y + 5), 3)
            clickable_buttons.append((tab_rect, f"tab_{tab}"))

        # Row 2: RL Architecture
        tab_y2 = tab_y1 + 24
        tab_w2 = (w - 20) // len(self.CATEGORY_RL)
        for i, tab in enumerate(self.CATEGORY_RL):
            tab_rect = pygame.Rect(x + 10 + i * tab_w2, tab_y2, tab_w2 - 2, 22)
            is_sel = (self.active_tab == tab)
            bg = T.C.accent if is_sel else T.C.panel_alt
            txt_col = T.C.text_on_accent if is_sel else T.C.text_dim
            pygame.draw.rect(surface, bg, tab_rect, border_radius=3)
            lbl = f_small.render(self.TAB_LABELS.get(tab, tab), True, txt_col)
            surface.blit(lbl, (tab_rect.centerx - lbl.get_width() // 2, tab_rect.centery - lbl.get_height() // 2))
            if tab in error_tabs:
                pygame.draw.circle(surface, T.C.error,
                                   (tab_rect.right - 6, tab_rect.y + 5), 3)
            clickable_buttons.append((tab_rect, f"tab_{tab}"))

        # Separator line
        sep_y = tab_y2 + 26
        pygame.draw.line(surface, T.C.border, (x + 10, sep_y), (x + w - 10, sep_y), 1)

        # 3. Property Rows — scrollable, clipped region
        props = self.get_properties_for_active_tab()
        self._props_count = len(props)
        bot_y = y + h - 80
        self.props_area = pygame.Rect(x + 4, sep_y + 4, w - 8, bot_y - sep_y - 10)
        max_off = max(0, len(props) * 24 - self.props_area.h)
        scroll_off = max(0, min(max_off,
                                self.tab_scroll.get(self.active_tab, 0)))
        self.tab_scroll[self.active_tab] = scroll_off
        surface.set_clip(self.props_area)
        row_y = sep_y + 6 - scroll_off
        row_h = 24

        for p in props:
            if row_y + row_h < self.props_area.top:
                row_y += row_h
                continue
            if row_y > self.props_area.bottom:
                break

            lbl_color = T.C.accent_line if "---" in p.label else T.C.text_dim
            surface.blit(f_small.render(p.label, True, lbl_color), (x + 12, row_y + 3))

            if p.prop_type in ("float", "int"):
                val_str = f"{p.current_value:.2f}{p.unit}" if p.prop_type == "float" else f"{int(p.current_value)}{p.unit}"
                lbl_val = f_mono.render(val_str, True, T.C.text)
                surface.blit(lbl_val, (x + w - 105 - lbl_val.get_width(), row_y + 3))

                btn_minus = pygame.Rect(x + w - 85, row_y + 1, 35, 19)
                btn_plus = pygame.Rect(x + w - 46, row_y + 1, 35, 19)
                pygame.draw.rect(surface, T.C.panel_alt, btn_minus, border_radius=3)
                pygame.draw.rect(surface, T.C.panel_alt, btn_plus, border_radius=3)
                surface.blit(f_bold.render("-", True, T.C.text), (btn_minus.centerx - 4, btn_minus.centery - 7))
                surface.blit(f_bold.render("+", True, T.C.text), (btn_plus.centerx - 5, btn_plus.centery - 7))

                clickable_buttons.append((btn_minus, f"prop_minus_{p.prop_id}"))
                clickable_buttons.append((btn_plus, f"prop_plus_{p.prop_id}"))

            elif p.prop_type == "bool":
                btn_toggle = pygame.Rect(x + w - 75, row_y + 1, 65, 19)
                bg_col = T.C.ok if p.current_value else T.C.panel_alt
                txt = "TRUE" if p.current_value else "FALSE"
                pygame.draw.rect(surface, bg_col, btn_toggle, border_radius=3)
                lbl_t = f_small.render(txt, True, T.C.text_on_accent)
                surface.blit(lbl_t, (btn_toggle.centerx - lbl_t.get_width() // 2, btn_toggle.centery - lbl_t.get_height() // 2))
                clickable_buttons.append((btn_toggle, f"prop_toggle_{p.prop_id}"))

            elif p.prop_type == "enum":
                btn_enum = pygame.Rect(x + w - 105, row_y + 1, 95, 19)
                pygame.draw.rect(surface, T.C.panel_alt, btn_enum, border_radius=3)
                lbl_e = f_small.render(str(p.current_value).upper()[:12], True, T.C.accent_line)
                surface.blit(lbl_e, (btn_enum.centerx - lbl_e.get_width() // 2, btn_enum.centery - lbl_e.get_height() // 2))
                clickable_buttons.append((btn_enum, f"prop_enum_{p.prop_id}"))

            elif p.prop_type == "action":
                btn_act = pygame.Rect(x + w - 130, row_y + 1, 120, 19)
                color = T.C.error if "DEL" in p.prop_id else T.C.accent
                pygame.draw.rect(surface, color, btn_act, border_radius=3)
                lbl_a = f_small.render(str(p.current_value), True, T.C.text_on_accent)
                surface.blit(lbl_a, (btn_act.centerx - lbl_a.get_width() // 2, btn_act.centery - lbl_a.get_height() // 2))
                clickable_buttons.append((btn_act, f"prop_act_{p.prop_id}"))

            elif p.prop_type == "chart":
                # Compact sparkline for metric series (e.g. episode rewards)
                series = [float(v) for v in (p.current_value or []) if isinstance(v, (int, float))]
                chart = pygame.Rect(x + w - 150, row_y + 2, 138, 18)
                pygame.draw.rect(surface, T.C.canvas, chart, border_radius=2)
                if len(series) >= 2:
                    lo, hi = min(series), max(series)
                    span = (hi - lo) or 1.0
                    pts = []
                    for i, v in enumerate(series[-60:]):
                        px = chart.x + 2 + i * (chart.w - 4) / max(1, len(series[-60:]) - 1)
                        py = chart.bottom - 2 - (v - lo) / span * (chart.h - 4)
                        pts.append((px, py))
                    pygame.draw.lines(surface, T.C.accent_line, False, pts, 1)

            elif p.prop_type == "multichart":
                # Multi-series comparison chart on a SHARED scale — the point
                # of a comparison view is a common axis.
                chart = pygame.Rect(x + w - 150, row_y + 2, 138, 18)
                pygame.draw.rect(surface, T.C.canvas, chart, border_radius=2)
                series_list = [
                    [float(v) for v in (s.get("data") or [])
                     if isinstance(v, (int, float))]
                    for s in (p.current_value or []) if isinstance(s, dict)
                ]
                flat = [v for s in series_list for v in s]
                if flat:
                    lo, hi = min(flat), max(flat)
                    span = (hi - lo) or 1.0
                    # Axis frame + min/max ticks.
                    pygame.draw.line(surface, T.C.separator,
                                     (chart.left, chart.bottom),
                                     (chart.right, chart.bottom), 1)
                    hi_txt = f_small.render(f"{hi:.0f}", True, T.C.text_faint)
                    lo_txt = f_small.render(f"{lo:.0f}", True, T.C.text_faint)
                    surface.blit(hi_txt, (chart.left - hi_txt.get_width() - 2,
                                          chart.top - 3))
                    surface.blit(lo_txt, (chart.left - lo_txt.get_width() - 2,
                                          chart.bottom - lo_txt.get_height() + 1))
                    colors = [(0, 210, 255), (255, 170, 60), (120, 255, 120),
                              (255, 100, 180)]
                    for si, vals in enumerate(series_list[:4]):
                        if len(vals) < 2:
                            continue
                        vals = vals[-60:]
                        pts = []
                        for i, v in enumerate(vals):
                            px = chart.x + 2 + i * (chart.w - 4) / max(1, len(vals) - 1)
                            py = chart.bottom - 2 - (v - lo) / span * (chart.h - 4)
                            pts.append((px, py))
                        pygame.draw.lines(
                            surface, colors[si % len(colors)], False, pts, 1)

            elif p.prop_type == "nav":
                # Clickable navigation link — jumps to the owning tab
                nav_rect = pygame.Rect(x + 10, row_y + 1, w - 20, 20)
                pygame.draw.rect(surface, T.C.panel_alt, nav_rect, border_radius=3)
                msg = str(p.current_value)
                lbl_n = f_small.render(msg[:40], True, T.C.info)
                surface.blit(lbl_n, (nav_rect.x + 6, nav_rect.y + 3))
                arrow = f_small.render(">", True, T.C.accent_line)
                surface.blit(arrow, (nav_rect.right - 14, nav_rect.y + 3))
                clickable_buttons.append((nav_rect, f"prop_nav_{p.prop_id}"))

            elif p.prop_type == "label":
                lbl_v = f_small.render(str(p.current_value), True, T.C.text_dim)
                surface.blit(lbl_v, (x + w - 12 - lbl_v.get_width(), row_y + 3))

            row_y += row_h

        surface.set_clip(None)
        # Scrollbar indicator when content overflows
        if max_off > 0:
            frac = self.props_area.h / max(1, len(props) * 24)
            bar_h = max(20, int(self.props_area.h * frac))
            rel = scroll_off / max_off
            bar_y = self.props_area.y + int((self.props_area.h - bar_h) * rel)
            pygame.draw.rect(surface, T.C.separator,
                             (self.props_area.right - 3, bar_y, 3, bar_h),
                             border_radius=2)

        # 4. Bottom Action Buttons (Rebuild 3D, New, Save, Load)
        bot_y = y + h - 80
        pygame.draw.line(surface, T.C.border, (x + 10, bot_y - 6), (x + w - 10, bot_y - 6), 1)

        # Rebuild 3D Mesh
        btn_rebuild = pygame.Rect(x + 12, bot_y, w - 24, 30)
        pygame.draw.rect(surface, T.C.ok, btn_rebuild, border_radius=4)
        lbl_rb = f_bold.render("Rebuild 3D", True, T.C.text_on_accent)
        surface.blit(lbl_rb, (btn_rebuild.centerx - lbl_rb.get_width() // 2, btn_rebuild.centery - lbl_rb.get_height() // 2))
        clickable_buttons.append((btn_rebuild, "action_rebuild_mesh"))

        # Save & Load & New Project Row
        sub_y = bot_y + 36
        sub_w = (w - 24 - 10) // 3
        b_new = pygame.Rect(x + 12, sub_y, sub_w, 26)
        b_save = pygame.Rect(x + 12 + sub_w + 5, sub_y, sub_w, 26)
        b_load = pygame.Rect(x + 12 + (sub_w + 5) * 2, sub_y, sub_w, 26)

        pygame.draw.rect(surface, T.C.panel_alt, b_new, border_radius=3)
        pygame.draw.rect(surface, T.C.accent, b_save, border_radius=3)
        pygame.draw.rect(surface, T.C.panel_alt, b_load, border_radius=3)

        surface.blit(f_small.render("New", True, T.C.text), (b_new.centerx - 12, b_new.centery - 6))
        surface.blit(f_small.render("Save", True, T.C.text_on_accent), (b_save.centerx - 14, b_save.centery - 6))
        surface.blit(f_small.render("Open...", True, T.C.text), (b_load.centerx - 14, b_load.centery - 6))

        clickable_buttons.append((b_new, "action_new_project"))
        clickable_buttons.append((b_save, "action_save_project"))
        clickable_buttons.append((b_load, "action_load_project"))

        return clickable_buttons
