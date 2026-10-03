"""TrackRegistry + EnvFactory tests."""
from __future__ import annotations

import numpy as np
import pytest


def test_registry_lists_tracks():
    from agentRL.envs.track_registry import TrackRegistry

    reg = TrackRegistry.default()
    ids = reg.list()
    assert "oval" in ids and "serpentine" in ids and "smoke" in ids


def test_registry_generated_tracks():
    from agentRL.envs.track_registry import TrackRegistry

    reg = TrackRegistry.default()
    gen = [t for t in reg.list() if t.startswith("gen_")]
    assert len(gen) >= 4, "need >=4 generated closed tracks for continual+holdout"
    a = reg.load(gen[0])
    assert a.project["road_definition"]["is_closed"] is True
    assert len(a.project["road_definition"]["control_points"]) >= 6


def test_factory_builds_env(tracks):
    from agentRL.envs.factory import EnvFactory
    from agentRL.envs.track_registry import TrackRegistry
    from agentRL.obs.spec import ObservationSpec, PRESETS

    factory = EnvFactory(obs_spec=ObservationSpec(channel_names=PRESETS["full23"]))
    env = factory.build(TrackRegistry.default().load("oval"), seed=42)
    obs, info = env.reset(seed=42)
    obs = np.asarray(obs, dtype=np.float32)
    assert obs.shape == (23,)
    assert np.isfinite(obs).all()


def test_reward_preset_installed(tracks):
    from agentRL.envs.factory import EnvFactory
    from agentRL.envs.track_registry import TrackRegistry

    factory = EnvFactory(reward="drive_v1")
    env = factory.build(TrackRegistry.default().load("oval"), seed=42)
    env.reset(seed=42)
    obs, r, term, trunc, info = env.step([0.0, 0.5, 0.0])
    keys = set(info["reward_breakdown"].keys())
    for cid in ("progress", "speed", "time_penalty", "centering"):
        assert cid in keys, f"missing reward component {cid}"


def test_obs_channel_subset(tracks):
    from agentRL.envs.factory import EnvFactory
    from agentRL.envs.track_registry import TrackRegistry
    from agentRL.obs.spec import ObservationSpec, PRESETS

    factory = EnvFactory(
        obs_spec=ObservationSpec(channel_names=PRESETS["state8"]))
    env = factory.build(TrackRegistry.default().load("oval"), seed=42)
    obs, _ = env.reset(seed=42)
    assert np.asarray(obs).shape == (8,)


def test_deterministic_reset(tracks):
    from agentRL.envs.factory import EnvFactory
    from agentRL.envs.track_registry import TrackRegistry

    factory = EnvFactory()
    reg = TrackRegistry.default()
    e1 = factory.build(reg.load("oval"), seed=7)
    e2 = factory.build(reg.load("oval"), seed=7)
    o1, _ = e1.reset(seed=99)
    o2, _ = e2.reset(seed=99)
    assert np.allclose(np.asarray(o1), np.asarray(o2))


def test_generated_track_builds_env(tracks):
    from agentRL.envs.factory import EnvFactory
    from agentRL.envs.track_registry import TrackRegistry

    factory = EnvFactory()
    env = factory.build(TrackRegistry.default().load("gen_loop_0"), seed=1)
    obs, _ = env.reset(seed=1)
    assert np.isfinite(np.asarray(obs)).all()


def test_unknown_track_raises():
    from agentRL.envs.track_registry import TrackRegistry

    with pytest.raises(KeyError):
        TrackRegistry.default().load("no_such_track")
