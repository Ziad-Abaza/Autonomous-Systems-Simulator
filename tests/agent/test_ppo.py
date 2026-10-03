"""PPOAgent tests — rollout, GAE/truncation bootstrap, update, checkpoint."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def _agent(**extra):
    from agentRL.algos.ppo import PPOAgent
    from agentRL.core.config import AgentConfig
    from agentRL.obs.spec import ObservationSpec, PRESETS

    spec = ObservationSpec(channel_names=PRESETS["state8"])
    cfg = AgentConfig(algo_id="ppo", hidden_sizes=(64, 64), lr=3e-4,
                      gamma=0.99, extra={"seed": 0, "steps_per_rollout": 32,
                                         **extra})
    act_space = {"low": np.array([-1.0, 0.0, 0.0]),
                 "high": np.array([1.0, 1.0, 1.0])}
    return PPOAgent(obs_spec=spec, act_space=act_space, cfg=cfg)


def _obs(seed=None):
    rng = np.random.default_rng(seed) if seed is not None else np.random
    return rng.uniform(-1, 1, size=8).astype(np.float32)


def _trans(agent, term=False, trunc=False, env_id=0, seed=None):
    obs = _obs(seed)
    a, aux = agent.act(obs)
    info = dict(aux)
    info["env_id"] = env_id
    agent.observe(obs, a, 1.0, _obs(),
                  terminated=term, truncated=trunc, info=info)


def test_act_bounds():
    agent = _agent()
    for _ in range(50):
        a, aux = agent.act(_obs())
        assert np.abs(a).max() <= 1.0 + 1e-6
        assert "logp" in aux and "value" in aux


def test_rollout_buffer_fills():
    agent = _agent()
    for _ in range(10):
        _trans(agent)
    assert len(agent.rollout["obs"]) == 10


def test_update_after_rollout():
    agent = _agent()
    before = [p.clone() for p in agent.actor.parameters()]
    for i in range(32):
        _trans(agent, term=(i == 20))
    losses = agent.update(32)
    assert losses and np.isfinite(losses["actor_loss"])
    changed = any(not torch.equal(p, b) for p, b in
                  zip(agent.actor.parameters(), before))
    assert changed, "update ran but actor params unchanged"
    assert len(agent.rollout["obs"]) == 0, "rollout not cleared"


def test_gae_truncation_bootstrap():
    """Truncated end bootstraps V(next_obs); terminated end does not."""
    agent = _agent()
    obs, nobs = _obs(seed=1), _obs(seed=2)
    a, aux = agent.act(obs)
    agent.observe(obs, a, 0.0, nobs, terminated=False, truncated=True,
                  info={**aux, "env_id": 0})
    obs_t = torch.as_tensor(obs).unsqueeze(0)
    nobs_t = torch.as_tensor(nobs).unsqueeze(0)
    with torch.no_grad():
        v_s = agent.critic(obs_t).item()
        v_next = agent.critic(nobs_t).item()
    adv_t, _ = agent._compute_gae()
    expected_trunc = agent.gamma * v_next - v_s  # bootstrapped
    assert adv_t[0] == pytest.approx(expected_trunc, abs=1e-4), (
        "truncated step did not bootstrap next value")

    agent2 = _agent()
    a2, aux2 = agent2.act(obs)
    agent2.observe(obs, a2, 0.0, nobs, terminated=True, truncated=False,
                   info={**aux2, "env_id": 0})
    obs2 = torch.as_tensor(obs).unsqueeze(0)
    with torch.no_grad():
        v_s2 = agent2.critic(obs2).item()
    adv2, _ = agent2._compute_gae()
    assert adv2[0] == pytest.approx(-v_s2, abs=1e-4), (
        "terminated step wrongly bootstrapped")


def test_checkpoint_roundtrip(tmp_path):
    agent = _agent()
    obs = _obs(seed=7)
    before, _ = agent.act(obs, deterministic=True)
    path = tmp_path / "ppo.pt"
    agent.save(str(path))
    from agentRL.algos.ppo import PPOAgent
    loaded = PPOAgent.load(str(path))
    after, _ = loaded.act(obs, deterministic=True)
    assert np.allclose(before, after)
