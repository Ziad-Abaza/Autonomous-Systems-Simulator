"""
Unit tests for Environment Template System, template customization,
and Training Configuration Exporter.
"""

import pytest
import os
import json
import tempfile

from sim_env.templates import EnvironmentTemplateManager
from sim_env.export import TrainingExporter
from sim_env.experiment import ExperimentConfig
from sim_project.serializer import EnvironmentProject


def test_template_instantiation():
    templates = EnvironmentTemplateManager.list_templates()
    assert len(templates) >= 4
    tmpl_ids = [t["id"] for t in templates]
    assert "empty" in tmpl_ids
    assert "basic_driving" in tmpl_ids
    assert "lane_following" in tmpl_ids
    assert "obstacle_avoidance" in tmpl_ids

    # Create basic driving
    proj_basic = EnvironmentTemplateManager.create_project_from_template("basic_driving")
    assert isinstance(proj_basic, EnvironmentProject)
    assert proj_basic.agent is not None
    assert len(proj_basic.road_def.control_points) > 0

    # Create lane following
    proj_lane = EnvironmentTemplateManager.create_project_from_template("lane_following")
    c_center = proj_lane.agent.reward_function.get_component("centering")
    assert c_center is not None
    assert c_center.weight == 1.0

    # Create obstacle avoidance
    proj_obs = EnvironmentTemplateManager.create_project_from_template("obstacle_avoidance")
    assert len(proj_obs.entities) > 0


def test_templates_serialize_to_json():
    """Every template must produce a JSON-serializable project dict."""
    for t in EnvironmentTemplateManager.list_templates():
        proj = EnvironmentTemplateManager.create_project_from_template(t["id"])
        data = proj.to_dict()
        json.dumps(data)
        proj.compute_fingerprint()


def test_obstacle_avoidance_entity_positions():
    """Template entities must be placed at their intended positions, not origin."""
    proj = EnvironmentTemplateManager.create_project_from_template("obstacle_avoidance")
    positions = {(round(e.pos.x, 3), round(e.pos.y, 3)) for e in proj.entities}
    assert (45.0, 15.0) in positions
    assert (-40.0, 10.0) in positions
    assert (0.0, 40.0) in positions
    assert (5.0, 41.0) in positions
    for e in proj.entities:
        assert isinstance(e.name, str)


def test_training_bundle_export():
    proj = EnvironmentTemplateManager.create_project_from_template("basic_driving")
    exp = ExperimentConfig(
        experiment_id="test_exp",
        name="Unit Test Experiment",
        algorithm="PPO",
        total_timesteps=1000
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = TrainingExporter.export_training_bundle(
            output_dir=tmpdir,
            env_project=proj,
            experiment=exp
        )

        assert os.path.exists(bundle["environment"])
        assert os.path.exists(bundle["scenario"])
        assert os.path.exists(bundle["experiment"])
        assert os.path.exists(bundle["runner"])

        # Validate environment.json content
        with open(bundle["environment"], "r", encoding="utf-8") as f:
            env_json = json.load(f)
        assert env_json["name"] == proj.name
        assert "agent" in env_json
        assert "road_definition" in env_json

        # Validate experiment.json content
        with open(bundle["experiment"], "r", encoding="utf-8") as f:
            exp_json = json.load(f)
        assert exp_json["algorithm"] == "PPO"
        assert exp_json["total_timesteps"] == 1000

        # Validate runner python script
        with open(bundle["runner"], "r", encoding="utf-8") as f:
            code = f.read()
        assert "PPORunner" in code
        assert "SimulationEnvironment" in code
