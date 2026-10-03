"""
Tests for Observation Space Designer, schema generation, ordering,
shapes, dtypes, bounds, serialization, and strict security isolation (anti-leakage).
"""

import pytest
import numpy as np
import math

from sim_env.observation_designer import (
    ObservationSpaceDefinition,
    ObservationChannelConfig,
    ChannelCategory,
    NormalizationType,
    CompiledObservationPipeline
)


def test_observation_space_default_generation():
    obs_space = ObservationSpaceDefinition.create_default_space()
    schema = obs_space.export_schema()

    assert schema["vector_dimension"] == 23  # 1+2+1+1+1+1+1+15
    assert schema["flatten_vector"] is True
    assert len(schema["channels"]) == 8

    # Verify channel properties
    names = [c["name"] for c in schema["channels"]]
    assert "speed" in names
    assert "velocity_body" in names
    assert "yaw_rate" in names
    assert "steering_angle" in names
    assert "distance_from_center" in names
    assert "heading_error" in names
    assert "distance_to_checkpoint" in names
    assert "lidar_ranges" in names


def test_observation_channel_enable_disable_reorder():
    obs_space = ObservationSpaceDefinition.create_default_space()
    assert obs_space.compute_vector_dim() == 23

    # Disable lidar rays (15 dims)
    assert obs_space.enable_channel("lidar_ranges", enabled=False) is True
    assert obs_space.compute_vector_dim() == 8

    # Reorder channels
    obs_space.reorder_channels(["distance_from_center", "speed", "heading_error"])
    active = obs_space.get_active_channels()
    assert active[0].name == "distance_from_center"
    assert active[1].name == "speed"
    assert active[2].name == "heading_error"


def test_observation_serialization_roundtrip():
    obs_space = ObservationSpaceDefinition.create_default_space()
    obs_space.add_channel(ObservationChannelConfig(
        name="custom_temp",
        channel_type="scalar",
        shape=[1],
        range_low=[0.0],
        range_high=[100.0],
        normalization=NormalizationType.MIN_MAX,
        norm_params={"min": 0.0, "max": 100.0}
    ))

    data = obs_space.to_dict()
    restored = ObservationSpaceDefinition.from_dict(data)

    assert restored.compute_vector_dim() == 24
    assert any(c.name == "custom_temp" for c in restored.channels)
    assert restored.flatten_vector == obs_space.flatten_vector


def test_compiled_observation_pipeline_execution():
    obs_space = ObservationSpaceDefinition.create_default_space()
    pipeline = obs_space.compile_pipeline()

    # Mock sensor samples
    sensor_samples = {
        "vehicle_state": {
            "speed": 22.5,
            "vel_body": np.array([22.5, 0.0], dtype=np.float32),
            "yaw_rate": 1.5,
            "steering_angle": 0.3,
            "distance_from_center": 0.0,
            "heading_error": 0.0,
            "distance_to_checkpoint": 50.0,
            "road_width": 12.0
        },
        "lidar_rays": {
            "ranges_norm": np.ones(15, dtype=np.float32) * 0.8
        }
    }

    obs = pipeline.build_observation(sensor_samples)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (23,)
    assert obs.dtype == np.float32
    assert pytest.approx(obs[0], abs=1e-3) == 0.5   # 22.5 / 45.0
    assert pytest.approx(obs[1], abs=1e-3) == 0.5   # vx / 45.0
    assert pytest.approx(obs[2], abs=1e-3) == 0.0   # vy / 10.0
    assert pytest.approx(obs[3], abs=1e-3) == 0.5   # yaw_rate 1.5 / 3.0
    assert pytest.approx(obs[4], abs=1e-3) == 0.5   # steer 0.3 / 0.6
    assert pytest.approx(obs[7], abs=1e-3) == 0.5   # dist_to_cp 50 / 100
    assert np.allclose(obs[8:], 0.8)                 # LiDAR rays


def test_critical_observation_leakage_rejection():
    """
    CRITICAL OBSERVATION RULE:
    Asserts that debug and oracle telemetry cannot silently enter the agent observation space.
    """
    obs_space = ObservationSpaceDefinition.create_default_space()

    # Attempt to inject debug telemetry into agent observation
    debug_channel = ObservationChannelConfig(
        name="oracle_global_pos",
        channel_type="vector",
        shape=[3],
        category=ChannelCategory.DEBUG_TELEMETRY,
        source_sensor="oracle",
        source_key="pos",
        description="Privileged global coordinates"
    )
    obs_space.add_channel(debug_channel)

    is_safe, violations = obs_space.validate_no_leakage()
    assert is_safe is False
    assert len(violations) > 0
    assert "debug_telemetry" in violations[0]

    # Pipeline compilation must raise ValueError
    with pytest.raises(ValueError, match="Observation security validation failed"):
        obs_space.compile_pipeline()
