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
    CATEGORY_RL = ["AGENT", "OBS", "ACTION", "REWARD", "TERM", "SCENARIO", "VALIDATE", "TRAIN"]
    ALL_TABS = CATEGORY_GEO + CATEGORY_RL

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

    def run_validation(self) -> ValidationReport:
        sensor_names = list(self.sensor_manager.sensors.keys()) if self.sensor_manager else []
        self.last_validation_report = EnvironmentValidator.validate(
            road_def=self.road_def,
            agent=self.agent,
            entities=self.entities,
            available_sensors=sensor_names
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
            props.append(PropertyRow("ov_val", "RL Readiness Gate", "label", status_txt))

            props.append(PropertyRow("ov_head_actions", "--- RL PIPELINE ACTIONS ---", "label", ""))
            props.append(PropertyRow("ov_btn_export", "Export Bundle", "action", "EXPORT TRAINING"))
            props.append(PropertyRow("ov_btn_tmpl_basic", "Template: Basic", "action", "LOAD BASIC"))
            props.append(PropertyRow("ov_btn_tmpl_lane", "Template: Lane Keep", "action", "LOAD LANE"))
            props.append(PropertyRow("ov_btn_tmpl_obs", "Template: Obstacles", "action", "LOAD OBSTACLES"))

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
                props.append(PropertyRow("val_gate", "Gate Status", "label", "PASS - READY" if rep.is_valid_for_rl else "FAIL - BLOCKED"))
                props.append(PropertyRow("val_errs", f"Errors ({len(rep.errors)})", "label", "None" if not rep.errors else rep.errors[0].message[:28]))
                props.append(PropertyRow("val_warns", f"Warnings ({len(rep.warnings)})", "label", f"{len(rep.warnings)} items"))
                props.append(PropertyRow("val_infos", f"Info ({len(rep.infos)})", "label", f"{len(rep.infos)} items"))
                props.append(PropertyRow("val_btn_run", "Re-Run Validator", "action", "VALIDATE NOW"))
                for err in rep.errors[:4]:
                    props.append(PropertyRow(f"val_err_{err.subsystem}", f"ERR: {err.subsystem}", "label", err.message[:30]))

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

            workers = data.get("workers") or []
            if workers:
                props.append(PropertyRow("trn_head_workers", "--- WORKERS ---", "label", ""))
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
            for i, ent in enumerate(self.entities[:8]):
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
            props.append(PropertyRow("track_closed", "Is Closed Loop", "bool", r.is_closed))
            props.append(PropertyRow("track_friction", "Surface Friction", "float", r.default_friction, 0.1, 2.5, 0.05))
            props.append(PropertyRow("track_checkpoints", "Checkpoints", "int", r.num_checkpoints, 4, 64, 2))
            props.append(PropertyRow("b_left_type", "Left Boundary", "enum", b.left_type, options=["guardrail", "wall", "curb", "open"]))
            props.append(PropertyRow("b_right_type", "Right Boundary", "enum", b.right_type, options=["guardrail", "wall", "curb", "open"]))
            props.append(PropertyRow("b_has_curbs", "Curb Ribbons", "bool", b.has_curbs))
            props.append(PropertyRow("b_curb_width", "Curb Width", "float", b.curb_width, 0.2, 2.0, 0.1, unit="m"))
            props.append(PropertyRow("b_wall_height", "Wall Height", "float", b.wall_height, 0.3, 3.0, 0.1, unit="m"))

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
                props.append(PropertyRow("cp_action_del", "Delete Point", "action", "DELETE POINT"))
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
                props.append(PropertyRow("ent_del", "Delete Entity", "action", "DELETE ENTITY"))
            else:
                props.append(PropertyRow("ent_none", "No Entity Selected", "label", "Click entity on canvas"))

        return props

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
        if prop_id.startswith(("obs_", "act_", "rf_", "term_", "ag_")) and self.on_agent_modified:
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
        pygame.draw.rect(surface, (16, 20, 28, 240), (x, y, w, h), border_radius=8)
        pygame.draw.rect(surface, (45, 55, 75), (x, y, w, h), 1, border_radius=8)

        f_bold = fonts['bold']
        f_small = fonts['small']
        f_mono = fonts['mono']
        clickable_buttons: List[Tuple[pygame.Rect, str]] = []

        # 1. Header
        header_y = y + 8
        surface.blit(f_bold.render("RL ENVIRONMENT DESIGNER", True, (0, 210, 255)), (x + 12, header_y))

        # 2. Two-tier category tab bar
        # Row 1: World & Geometry
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

        # Row 2: RL Architecture
        tab_y2 = tab_y1 + 24
        tab_w2 = (w - 20) // len(self.CATEGORY_RL)
        for i, tab in enumerate(self.CATEGORY_RL):
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
        row_h = 24

        for p in props:
            if row_y > (y + h - 85):
                break

            lbl_color = (0, 200, 255) if "---" in p.label else (180, 195, 210)
            surface.blit(f_small.render(p.label, True, lbl_color), (x + 12, row_y + 3))

            if p.prop_type in ("float", "int"):
                val_str = f"{p.current_value:.2f}{p.unit}" if p.prop_type == "float" else f"{int(p.current_value)}{p.unit}"
                lbl_val = f_mono.render(val_str, True, (255, 255, 255))
                surface.blit(lbl_val, (x + w - 105 - lbl_val.get_width(), row_y + 3))

                btn_minus = pygame.Rect(x + w - 85, row_y + 1, 35, 19)
                btn_plus = pygame.Rect(x + w - 46, row_y + 1, 35, 19)
                pygame.draw.rect(surface, (35, 45, 60), btn_minus, border_radius=3)
                pygame.draw.rect(surface, (35, 45, 60), btn_plus, border_radius=3)
                surface.blit(f_bold.render("-", True, (220, 220, 220)), (btn_minus.centerx - 4, btn_minus.centery - 7))
                surface.blit(f_bold.render("+", True, (220, 220, 220)), (btn_plus.centerx - 5, btn_plus.centery - 7))

                clickable_buttons.append((btn_minus, f"prop_minus_{p.prop_id}"))
                clickable_buttons.append((btn_plus, f"prop_plus_{p.prop_id}"))

            elif p.prop_type == "bool":
                btn_toggle = pygame.Rect(x + w - 75, row_y + 1, 65, 19)
                bg_col = (30, 140, 70) if p.current_value else (80, 40, 40)
                txt = "TRUE" if p.current_value else "FALSE"
                pygame.draw.rect(surface, bg_col, btn_toggle, border_radius=3)
                lbl_t = f_small.render(txt, True, (255, 255, 255))
                surface.blit(lbl_t, (btn_toggle.centerx - lbl_t.get_width() // 2, btn_toggle.centery - lbl_t.get_height() // 2))
                clickable_buttons.append((btn_toggle, f"prop_toggle_{p.prop_id}"))

            elif p.prop_type == "enum":
                btn_enum = pygame.Rect(x + w - 105, row_y + 1, 95, 19)
                pygame.draw.rect(surface, (35, 48, 68), btn_enum, border_radius=3)
                lbl_e = f_small.render(str(p.current_value).upper()[:12], True, (0, 210, 255))
                surface.blit(lbl_e, (btn_enum.centerx - lbl_e.get_width() // 2, btn_enum.centery - lbl_e.get_height() // 2))
                clickable_buttons.append((btn_enum, f"prop_enum_{p.prop_id}"))

            elif p.prop_type == "action":
                btn_act = pygame.Rect(x + w - 130, row_y + 1, 120, 19)
                color = (150, 40, 40) if "DEL" in p.prop_id else (0, 110, 170)
                pygame.draw.rect(surface, color, btn_act, border_radius=3)
                lbl_a = f_small.render(str(p.current_value), True, (255, 255, 255))
                surface.blit(lbl_a, (btn_act.centerx - lbl_a.get_width() // 2, btn_act.centery - lbl_a.get_height() // 2))
                clickable_buttons.append((btn_act, f"prop_act_{p.prop_id}"))

            elif p.prop_type == "chart":
                # Compact sparkline for metric series (e.g. episode rewards)
                series = [float(v) for v in (p.current_value or []) if isinstance(v, (int, float))]
                chart = pygame.Rect(x + w - 150, row_y + 2, 138, 18)
                pygame.draw.rect(surface, (20, 26, 36), chart, border_radius=2)
                if len(series) >= 2:
                    lo, hi = min(series), max(series)
                    span = (hi - lo) or 1.0
                    pts = []
                    for i, v in enumerate(series[-60:]):
                        px = chart.x + 2 + i * (chart.w - 4) / max(1, len(series[-60:]) - 1)
                        py = chart.bottom - 2 - (v - lo) / span * (chart.h - 4)
                        pts.append((px, py))
                    pygame.draw.lines(surface, (0, 210, 255), False, pts, 1)

            elif p.prop_type == "multichart":
                # Multi-series comparison chart on a SHARED scale — the point
                # of a comparison view is a common axis.
                chart = pygame.Rect(x + w - 150, row_y + 2, 138, 18)
                pygame.draw.rect(surface, (20, 26, 36), chart, border_radius=2)
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
                    pygame.draw.line(surface, (60, 70, 90),
                                     (chart.left, chart.bottom),
                                     (chart.right, chart.bottom), 1)
                    hi_txt = f_small.render(f"{hi:.0f}", True, (110, 120, 140))
                    lo_txt = f_small.render(f"{lo:.0f}", True, (110, 120, 140))
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

            elif p.prop_type == "label":
                lbl_v = f_small.render(str(p.current_value), True, (150, 160, 175))
                surface.blit(lbl_v, (x + w - 12 - lbl_v.get_width(), row_y + 3))

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

        surface.blit(f_small.render("NEW", True, (230, 230, 230)), (b_new.centerx - 12, b_new.centery - 6))
        surface.blit(f_small.render("SAVE", True, (255, 255, 255)), (b_save.centerx - 14, b_save.centery - 6))
        surface.blit(f_small.render("LOAD", True, (230, 230, 230)), (b_load.centerx - 14, b_load.centery - 6))

        clickable_buttons.append((b_new, "action_new_project"))
        clickable_buttons.append((b_save, "action_save_project"))
        clickable_buttons.append((b_load, "action_load_project"))

        return clickable_buttons
