"""
Phase 7.5 — Core Functional Repair regression tests.

Covers the defects found in docs/PHASE_7_5_FUNCTIONAL_AUDIT.md:
A1/A2/A3 track semantics (open/closed, spawn-on-road validation)
B1 entity editing through the inspector
C1–C4 sensor/camera authoring, serialization, multi-camera observations
D1–D4 experiment paths and phantom-manifest filtering
"""

import math
import os
import json

import numpy as np
import pytest

from sim_env.templates import EnvironmentTemplateManager
from sim_env.validator import EnvironmentValidator
from sim_env.agent import AgentDefinition
from sim_env.sensor_config import SensorConfig, default_suite_configs
from sim_core.sensors.sensor_manager import SensorManager
from sim_core.math_utils import Vec3
from sim_ui.inspector import EnvironmentInspector


# ---------------------------------------------------------------- helpers

def _insp(proj):
    return EnvironmentInspector(proj.road_def, proj.vehicle_config,
                                SensorManager(), entities=proj.entities,
                                agent=proj.agent,
                                scenario_def=proj.scenario_def)


# ================================================================ TRACK

class TestTrackSemantics:
    def test_templates_all_validate(self):
        for t in EnvironmentTemplateManager.list_templates():
            proj = EnvironmentTemplateManager.create_project_from_template(t["id"])
            report = EnvironmentValidator.validate(
                proj.road_def, proj.agent, proj.entities)
            assert report.errors == [], \
                f"template {t['id']} has errors: {[i.message for i in report.errors]}"

    def test_spawn_off_road_is_error(self):
        proj = EnvironmentTemplateManager.create_project_from_template("basic_driving")
        proj.road_def.spawn_point.x = 500.0
        proj.road_def.spawn_point.y = 500.0
        report = EnvironmentValidator.validate(proj.road_def, proj.agent, proj.entities)
        assert any(i.subsystem == "Spawn" and i.severity == "ERROR"
                   for i in report.issues)

    def test_spawn_heading_misaligned_warns(self):
        proj = EnvironmentTemplateManager.create_project_from_template("basic_driving")
        # face backwards on a valid on-road point
        proj.road_def.spawn_point.x = 65.0
        proj.road_def.spawn_point.y = 0.0
        proj.road_def.spawn_point.yaw = math.pi  # reversed
        report = EnvironmentValidator.validate(proj.road_def, proj.agent, proj.entities)
        assert any(i.subsystem == "Spawn" and i.severity == "WARNING"
                   for i in report.issues)

    def test_open_track_reports_finish_semantics(self):
        proj = EnvironmentTemplateManager.create_project_from_template("empty")
        report = EnvironmentValidator.validate(proj.road_def, proj.agent, proj.entities)
        assert any("final checkpoint" in i.message for i in report.infos)

    def test_open_closed_round_trip(self):
        from sim_project.serializer import EnvironmentProject
        proj = EnvironmentTemplateManager.create_project_from_template("empty")
        d = proj.to_dict()
        proj2 = EnvironmentProject.from_dict(d)
        assert proj2.road_def.is_closed is False

    def test_editor_draw_closes_loop(self):
        """A1: clicking near cp0 of an open track with the draw tool closes it."""
        proj = EnvironmentTemplateManager.create_project_from_template("empty")
        from sim_ui.editor import VisualTrackEditor
        ed = VisualTrackEditor(proj.road_def, proj.entities)
        ed.tool = "draw"
        ed._needs_frame = False
        ed.camera_target = Vec3(cp0x := proj.road_def.control_points[0].x,
                                proj.road_def.control_points[0].y, 0)
        import pygame
        ed.canvas_rect = pygame.Rect(0, 0, 1200, 700)
        sx, sy = ed.world_to_screen(cp0x, proj.road_def.control_points[0].y)
        assert ed.handle_mouse_down((sx + 3, sy + 2), 1) is True
        assert proj.road_def.is_closed is True


# ================================================================ ENTITY

class TestEntityEditing:
    def test_entity_properties_mutate(self):
        proj = EnvironmentTemplateManager.create_project_from_template(
            "obstacle_avoidance")
        insp = _insp(proj)
        ent = proj.entities[0]
        insp.select_entity(ent.entity_id)
        x0, yaw0 = ent.pos.x, ent.yaw
        insp.handle_property_change("ent_pos_x", 5.0)
        insp.handle_property_change("ent_yaw", 45.0)
        assert ent.pos.x == pytest.approx(x0 + 5.0)
        assert ent.yaw == pytest.approx(yaw0 + math.radians(45.0))

    def test_entity_delete(self):
        proj = EnvironmentTemplateManager.create_project_from_template(
            "obstacle_avoidance")
        insp = _insp(proj)
        ent = proj.entities[0]
        insp.select_entity(ent.entity_id)
        insp.handle_property_change("ent_del", None)
        assert ent not in proj.entities
        assert insp.selected_entity_id is None

    def test_template_entities_have_readable_names(self):
        proj = EnvironmentTemplateManager.create_project_from_template(
            "obstacle_avoidance")
        for ent in proj.entities:
            assert "Vec3" not in ent.name

    def test_entity_edit_persists(self, tmp_path):
        from sim_project.serializer import EnvironmentProject
        proj = EnvironmentTemplateManager.create_project_from_template(
            "obstacle_avoidance")
        insp = _insp(proj)
        ent = proj.entities[0]
        insp.select_entity(ent.entity_id)
        insp.handle_property_change("ent_pos_x", 5.0)
        proj2 = EnvironmentProject.from_dict(proj.to_dict())
        ent2 = [e for e in proj2.entities if e.entity_id == ent.entity_id][0]
        assert ent2.pos.x == pytest.approx(ent.pos.x)


# ================================================================ SENSORS

class TestSensorConfigs:
    def test_agent_round_trip(self):
        a = AgentDefinition.create_default_vehicle_agent()
        a2 = AgentDefinition.from_dict(a.to_dict())
        assert [c.name for c in a2.sensor_configs] == \
            ["vehicle_state", "lidar_rays", "rgb_camera", "imu"]
        assert a2.sensor_names == ["vehicle_state", "lidar_rays",
                                   "rgb_camera", "imu"]

    def test_legacy_sensor_names_upgrade(self):
        a = AgentDefinition.from_dict({
            "agent_id": "x",
            "sensor_names": ["vehicle_state", "rgb_camera"],
        })
        assert [c.name for c in a.sensor_configs] == \
            ["vehicle_state", "rgb_camera"]
        assert a.sensor_configs[1].sensor_type == "camera_rgb"

    def test_manager_builds_from_configs(self):
        cfgs = default_suite_configs()
        cfgs.append(SensorConfig.for_type("camera_rgb", "rear_camera"))
        sm = SensorManager.build_from_configs(cfgs)
        assert "rear_camera" in sm.sensors
        assert sm.sensors["rear_camera"].sensor_type == "camera_rgb"

    def test_inspector_sensor_ops(self):
        proj = EnvironmentTemplateManager.create_project_from_template("basic_driving")
        insp = _insp(proj)
        insp.handle_property_change("sen_add_camera", None)
        added = proj.agent.sensor_configs[-1]
        assert added.sensor_type == "camera_rgb"
        assert added.name != "rgb_camera" or "rgb_camera" not in [
            c.name for c in proj.agent.sensor_configs[:-1]]
        insp.handle_property_change("sen_fov", 30.0)
        assert added.merged_params()["fov_degrees"] == pytest.approx(105.0)
        # image channel auto-registered for the new camera
        names = [s["name"] for s in
                 proj.agent.observation_space.image_channel_specs()]
        assert added.name in names

    def test_duplicate_names_are_validation_errors(self):
        proj = EnvironmentTemplateManager.create_project_from_template("basic_driving")
        proj.agent.sensor_configs.append(
            SensorConfig.for_type("camera_rgb", "rgb_camera"))
        report = EnvironmentValidator.validate(proj.road_def, proj.agent,
                                               proj.entities)
        assert any("Duplicate sensor" in i.message for i in report.errors)

    def test_multi_camera_observation_keys(self):
        a = AgentDefinition.create_default_vehicle_agent()
        a.observation_space.image_channels = [
            {"name": "rgb_camera", "shape": [84, 84, 3]},
            {"name": "rear_camera", "shape": [32, 32, 3]},
        ]
        pipe = a.observation_space.compile_pipeline()
        obs = pipe.build_observation({
            "vehicle_state": {"speed": 5.0},
            "rgb_camera": np.zeros((84, 84, 3), dtype=np.uint8),
            "rear_camera": np.ones((32, 32, 3), dtype=np.uint8) * 255,
        })
        assert set(obs.keys()) == {"vector", "image", "image_rear_camera"}
        assert obs["image"].shape == (84, 84, 3)
        assert obs["image_rear_camera"].shape == (32, 32, 3)
        assert obs["image_rear_camera"].max() == 255

    def test_legacy_include_image_channel(self):
        a = AgentDefinition.create_default_vehicle_agent()
        a.observation_space.include_image_channel = True
        pipe = a.observation_space.compile_pipeline()
        obs = pipe.build_observation({"vehicle_state": {"speed": 5.0}})
        assert "image" in obs and obs["image"].shape == (84, 84, 3)

    def test_env_builds_authored_suite(self):
        from sim_env.environment import SimulationEnvironment
        from sim_core.track.road_definition import RoadDefinition
        road = RoadDefinition.create_default_oval()
        a = AgentDefinition.create_default_vehicle_agent()
        a.sensor_configs = default_suite_configs()
        a.sensor_configs.append(SensorConfig.for_type("camera_rgb", "rear_camera"))
        env = SimulationEnvironment(road_def=road, agent=a)
        assert "rear_camera" in env.sensors.sensors


# ================================================================ EXPERIMENTS

class TestExperimentPaths:
    def test_legacy_manifest_filtered(self, tmp_path):
        from sim_experiment.manager import ExperimentManager
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        # legacy non-manifest experiment.json
        leg = tmp_path / "experiments" / "baseline_ui_env"
        leg.mkdir(parents=True)
        (leg / "experiment.json").write_text(json.dumps({
            "experiment_id": "exp_lane_keeping_01",
            "name": "ppo_ui_baseline_oval"}))
        assert mgr.list_experiments() == []

    def test_contract_paths_are_relative_and_resolve(self, tmp_path):
        from sim_experiment.manager import ExperimentManager
        from sim_experiment.manifest import ExperimentManifest
        from sim_experiment.run import RunManager
        from sim_experiment.trainer_contract import (
            build_contract, validate_contract, resolve_contract_paths)
        from sim_env.templates import EnvironmentTemplateManager
        from sim_experiment.manifest import TrainingConfig, EvaluationConfig

        proj = EnvironmentTemplateManager.create_project_from_template("basic_driving")
        manifest = ExperimentManifest.from_project(
            project=proj, scenario=proj.scenario_def,
            training=TrainingConfig(algorithm="ppo", total_timesteps=100),
            evaluation=EvaluationConfig(eval_seeds=[0], num_episodes=1),
            name="t", random_seed=1)
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        exp_dir = mgr.create(manifest)
        rm = RunManager()
        run = rm.create_run(exp_dir, seed=1)
        rd = rm.run_dir(exp_dir, run.run_id)
        contract = build_contract(manifest, exp_dir, rd, run.run_id)
        assert validate_contract(contract) == []
        # portable: no absolute paths baked in
        for v in contract["paths"].values():
            assert not os.path.isabs(v), v
        # resolution restores absolute paths for the runtime
        resolved = resolve_contract_paths(contract, rd)
        assert os.path.isabs(resolved["paths"]["metrics_file"])
        assert resolved["paths"]["run_dir"] == os.path.abspath(rd)


# ================================================================ WIDGETS

class TestTextInput:
    def test_caret_stays_in_bounds(self):
        import pygame
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        from sim_ui.widgets import UIContext, Fonts, text_input
        surf = pygame.Surface((400, 100))
        ctx = UIContext(surf, Fonts())
        iid = "long_name"
        ctx.focused_input = iid
        ctx.inputs[iid] = {"text": "x" * 300, "caret": 300}
        r = pygame.Rect(10, 10, 200, 30)
        text_input(ctx, iid, r)
        scroll = ctx.inputs[iid]["scroll"]
        caret_px = ctx.fonts.body.size("x" * 300)[0]
        assert caret_px - scroll <= r.w - 16
        pygame.quit()
