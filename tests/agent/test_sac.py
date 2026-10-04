"""SACAgent tests — bounds, update, checkpoint, truncation bootstrap, NaN guard."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def _agent(**extra):
    from agentRL.algos.sac import SACAgent
    from agentRL.core.config import AgentConfig
    from agentRL.obs.spec import ObservationSpec, PRESETS

    spec = ObservationSpec(channel_names=PRESETS["state8"])
    cfg = AgentConfig(algo_id="sac", hidden_sizes=(64, 64), lr=3e-4,
                      gamma=0.99, extra={"seed": 0, "warmup": 32,
                                         "batch": 16, **extra})
    act_space = {"low": np.array([-1.0, 0.0, 0.0]),
                 "high": np.array([1.0, 1.0, 1.0])}
    return SACAgent(obs_spec=spec, act_space=act_space, cfg=cfg)


def _obs(n=8, seed=None):
    rng = np.random.default_rng(seed) if seed else np.random
    return rng.uniform(-1, 1, size=n).astype(np.float32)


def test_act_bounds():
    agent = _agent()
    for _ in range(100):
        a, _ = agent.act(_obs(), deterministic=False)
        assert np.abs(a).max() <= 1.0 + 1e-6
    a, _ = agent.act(_obs(), deterministic=True)
    assert np.abs(a).max() <= 1.0 + 1e-6


def test_deterministic_act_repeatable():
    agent = _agent()
    obs = _obs(seed=1)
    a1, _ = agent.act(obs, deterministic=True)
    a2, _ = agent.act(obs, deterministic=True)
    assert np.array_equal(a1, a2)


def test_observe_and_update_smoke():
    agent = _agent()
    for i in range(64):
        agent.observe(_obs(), np.zeros(3), 1.0, _obs(),
                      terminated=False, truncated=False, info={})
    losses = agent.update(64)
    assert losses, "update should return losses post-warmup"
    for k, v in losses.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_checkpoint_roundtrip(tmp_path):
    agent = _agent()
    obs = _obs(seed=5)
    before, _ = agent.act(obs, deterministic=True)
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))
    loaded = agent.load(str(path))
    after, _ = loaded.act(obs, deterministic=True)
    assert np.allclose(before, after)
    assert loaded.train_state["steps"] == agent.train_state["steps"]


def test_pos_only_brake_survives_roundtrip(tmp_path):
    """Brake semantics must serialize — a loaded agent whose adapter
    silently reverts to affine parks the car (raw 0 -> brake 0.5)."""
    from agentRL.algos.sac import SACAgent
    from agentRL.core.config import AgentConfig
    from agentRL.obs.spec import ObservationSpec, PRESETS
    spec = ObservationSpec(channel_names=PRESETS["state8"])
    cfg = AgentConfig(algo_id="sac", hidden_sizes=(32, 32), lr=3e-4,
                      gamma=0.99, extra={"seed": 0, "warmup": 8})
    act_space = {"low": np.array([-1.0, 0.0, 0.0]),
                 "high": np.array([1.0, 1.0, 1.0]),
                 "pos_only": (2,)}
    agent = SACAgent(obs_spec=spec, act_space=act_space, cfg=cfg)
    path = tmp_path / "ck.pt"
    agent.save(str(path))
    loaded = SACAgent.load(str(path))
    assert loaded.act_space.get("pos_only") == (2,)
    # raw brake = 0 must stay released after reload
    env_a = loaded.act_space_to_env(np.array([0.0, 0.0, 0.0]))
    assert env_a[2] == 0.0


def test_truncation_bootstrap():
    """Q target must bootstrap on truncation, not zero out."""
    agent = _agent()
    agent.memory.begin_track("default")
    obs = _obs(seed=2)
    next_obs = _obs(seed=3)
    act_env = agent.act_space_to_env(np.zeros(3))
    # same transition twice: once terminated, once truncated
    agent.observe(obs, act_env, 1.0, next_obs,
                  terminated=True, truncated=False, info={})
    agent.observe(obs, act_env, 1.0, next_obs,
                  terminated=False, truncated=True, info={})
    assert agent.memory.buffers["default"].done[0] == 1.0
    assert agent.memory.buffers["default"].done[1] == 0.0


def test_nan_guard():
    agent = _agent()
    agent.memory.begin_track("default")
    obs = _obs(seed=4)
    act_env = agent.act_space_to_env(np.zeros(3))
    for _ in range(40):
        agent.observe(obs, act_env, np.nan, _obs(),
                      terminated=False, truncated=False, info={})
    params_before = [p.clone() for p in agent.actor.parameters()]
    losses = agent.update(40)
    assert losses.get("nan_guard", 0) >= 1
    for p, pb in zip(agent.actor.parameters(), params_before):
        assert torch.equal(p, pb), "params changed despite NaN batch"


def test_version_refusal(tmp_path):
    from agentRL.algos.sac import SACAgent
    from agentRL.core.versioning import VersionError

    agent = _agent()
    path = tmp_path / "ck.pt"
    agent.save(str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 999
    torch.save(payload, path)
    with pytest.raises(VersionError):
        SACAgent.load(str(path))
