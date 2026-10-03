"""
Unit tests for Action Space Designer, validation engine, dead zones,
rate limiting, continuous/discrete decoding, and malformed input handling.
"""

import pytest
import numpy as np

from sim_env.action_designer import (
    ActionSpaceDefinition,
    ActionChannelConfig,
    ActionType,
    DiscreteActionOption,
    CompiledActionDecoder
)


def test_action_space_continuous_defaults():
    act_space = ActionSpaceDefinition.create_default_vehicle_action_space(continuous=True)
    schema = act_space.export_schema()

    assert schema["space_type"] == ActionType.CONTINUOUS
    assert schema["num_channels"] == 3
    channels = {c["name"]: c for c in schema["channels"]}
    assert "steering" in channels
    assert "throttle" in channels
    assert "brake" in channels

    assert channels["steering"]["min"] == -1.0
    assert channels["steering"]["max"] == 1.0
    assert channels["throttle"]["min"] == 0.0
    assert channels["throttle"]["max"] == 1.0


def test_action_space_discrete_defaults():
    act_space = ActionSpaceDefinition.create_default_vehicle_action_space(continuous=False)
    schema = act_space.export_schema()

    assert schema["space_type"] == ActionType.DISCRETE
    assert schema["num_discrete_actions"] == 5
    decoder = act_space.compile_decoder()

    # Test discrete decoding: 0=Coast, 1=Accelerate, 2=Brake
    d0 = decoder.decode(0)
    assert np.allclose(d0, [0.0, 0.0, 0.0])

    d1 = decoder.decode(1)
    assert np.allclose(d1, [0.0, 0.7, 0.0])

    d2 = decoder.decode(2)
    assert np.allclose(d2, [0.0, 0.0, 0.8])


def test_action_validation_defensive_checks():
    act_space = ActionSpaceDefinition.create_default_vehicle_action_space(continuous=True)
    decoder = act_space.compile_decoder()

    # 1. None action
    ok, err = decoder.validate_action(None)
    assert ok is False
    assert "None" in err

    # 2. Empty action
    ok, err = decoder.validate_action([])
    assert ok is False
    assert "empty" in err

    # 3. NaN action
    ok, err = decoder.validate_action([np.nan, 0.5, 0.0])
    assert ok is False
    assert "NaN" in err

    # 4. Infinity action
    ok, err = decoder.validate_action([0.0, np.inf, 0.0])
    assert ok is False
    assert "Infinity" in err

    # 5. Missing dimension (< 3)
    ok, err = decoder.validate_action([0.5, 0.2])
    assert ok is False
    assert "less than" in err

    # 6. Valid action
    ok, err = decoder.validate_action([0.2, 0.8, 0.0])
    assert ok is True
    assert err == ""

    # Extra dimensions should safely validate or decode first 3
    ok, err = decoder.validate_action([0.1, 0.2, 0.3, 0.4])
    assert ok is True


def test_action_decoder_nan_inf_sanitization():
    act_space = ActionSpaceDefinition.create_default_vehicle_action_space(continuous=True)
    decoder = act_space.compile_decoder()

    # Even if malformed input is forced into decode, it must sanitize and never return NaN or Inf
    decoded = decoder.decode([np.nan, np.inf, -np.inf], dt=0.0)
    assert not np.any(np.isnan(decoded))
    assert not np.any(np.isinf(decoded))
    assert decoded[0] == 0.0  # NaN replaced by default
    assert decoded[1] == 1.0  # Inf clamped to max
    assert decoded[2] == 0.0  # -Inf clamped to min


def test_action_deadzone_and_rate_limiting():
    # Channel with dead zone 0.05 and rate limit 2.0 / sec
    channels = [
        ActionChannelConfig(name="steer", min_val=-1.0, max_val=1.0, default_val=0.0, dead_zone=0.05, rate_limit=2.0)
    ]
    space = ActionSpaceDefinition(space_type=ActionType.CONTINUOUS, channels=channels)
    decoder = space.compile_decoder()
    decoder.reset()

    # Small input inside dead zone (< 0.05) should snap to default (0.0)
    out_dz = decoder.decode([0.03], dt=0.0166)
    assert out_dz[0] == 0.0

    # Input outside dead zone (> 0.05)
    # Rate limit test: from 0.0 trying to jump to 1.0 with dt=0.1
    # Max delta = 2.0 * 0.1 = 0.2
    out_rl = decoder.decode([1.0], dt=0.1)
    assert pytest.approx(out_rl[0], abs=1e-3) == 0.2
