"""
Integration test running PPO Baseline against a UI-created Declarative Environment.
Verifies the complete pipeline:
  Environment Template -> Serialized Project -> Compiled Pipelines -> PPO Runner -> Convergence Metrics
"""

import os
import tempfile
import pytest
import numpy as np

from sim_env.templates import EnvironmentTemplateManager
from sim_env.environment import SimulationEnvironment
from sim_env.export import TrainingExporter
from sim_client.agents.ppo_baseline import PPORunner


def test_ppo_baseline_with_ui_created_environment():
    # 1. Instantiate declarative environment via template
    proj = EnvironmentTemplateManager.create_project_from_template("lane_following")
    assert proj.agent is not None
    assert proj.agent.observation_space.compute_vector_dim() == 23
    assert len(proj.agent.action_space.channels) == 3

    # 2. Export training bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = TrainingExporter.export_training_bundle(
            output_dir=tmpdir,
            env_project=proj,
            scenario=proj.scenario_def,
            experiment=proj.experiment_config
        )
        assert os.path.exists(bundle["environment"])
        assert os.path.exists(bundle["scenario"])
        assert os.path.exists(bundle["experiment"])
        assert os.path.exists(bundle["runner"])

    # 3. Instantiate simulation environment using compiled Phase 3 pipelines
    env = SimulationEnvironment(
        road_def=proj.road_def,
        vehicle_config=proj.vehicle_config,
        agent=proj.agent,
        episode_config=proj.episode_config,
        scenario_def=proj.scenario_def,
        seed=42
    )
    for ent in proj.entities:
        env.add_entity(ent)

    # 4. Train PPO baseline
    runner = PPORunner(
        env=env,
        num_steps=128,
        batch_size=64,
        num_epochs=2,
        seed=42
    )
    metrics = runner.train(total_timesteps=256)

    assert metrics["total_timesteps"] == 256
    assert metrics["sps"] > 50.0  # Must maintain high throughput (>50 SPS with PyTorch)
    assert len(metrics["policy_losses"]) == 2
    assert len(metrics["value_losses"]) == 2
