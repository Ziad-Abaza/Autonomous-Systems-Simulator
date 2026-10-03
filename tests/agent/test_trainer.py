"""Trainer + metrics + checkpoint-io tests."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest


def _sac(seed=0, **extra):
    from agentRL.algos.sac import SACAgent
    from agentRL.core.config import AgentConfig
    from agentRL.obs.spec import ObservationSpec, PRESETS

    spec = ObservationSpec(channel_names=PRESETS["state8"])
    cfg = AgentConfig(algo_id="sac", hidden_sizes=(64, 64), lr=3e-4,
                      gamma=0.99,
                      extra={"seed": seed, "warmup": 32, "batch": 16,
                             "random_steps": 16, **extra})
    return SACAgent(obs_spec=spec,
                    act_space={"low": np.array([-1.0, 0.0, 0.0]),
                               "high": np.array([1.0, 1.0, 1.0])},
                    cfg=cfg)


def _factory():
    from agentRL.envs.factory import EnvFactory
    from agentRL.obs.spec import ObservationSpec, PRESETS
    return EnvFactory(obs_spec=ObservationSpec(channel_names=PRESETS["state8"]))


def _track():
    from agentRL.envs.track_registry import TrackRegistry
    return TrackRegistry.default().load("smoke")


def test_metrics_jsonl(tmp_path):
    from agentRL.train.metrics import MetricsLogger

    log = MetricsLogger(str(tmp_path))
    log.episode(10, {"return": 5.0, "len": 10})
    log.update(20, {"loss": float("nan"), "q": 1.0})
    log.eval(30, {"mean_speed": 0.5})
    log.close()

    rows = [json.loads(l) for l in
            open(tmp_path / "metrics.jsonl", encoding="utf-8")]
    assert [r["scope"] for r in rows] == ["episode", "update", "eval"]
    assert rows[1]["metrics"]["loss"] is None  # NaN -> null
    assert [r["seq"] for r in rows] == sorted(r["seq"] for r in rows)


def test_trainer_smoke(tmp_path):
    from agentRL.core.config import TrainConfig
    from agentRL.train.trainer import OffPolicyTrainer

    cfg = TrainConfig(total_steps=200, eval_interval=10**9,
                      ckpt_interval=200, num_envs=1, seed=42,
                      run_dir=str(tmp_path), eval_episodes=1)
    tr = OffPolicyTrainer(_factory(), _track(), _sac(), cfg)
    summary = tr.train()
    assert (tmp_path / "metrics.jsonl").exists()
    rows = [json.loads(l) for l in
            open(tmp_path / "metrics.jsonl", encoding="utf-8")]
    scopes = {r["scope"] for r in rows}
    assert "heartbeat" in scopes
    assert summary["timesteps"] == 200
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_step_after_done_guard(tmp_path):
    from agentRL.core.config import TrainConfig
    from agentRL.train.trainer import OffPolicyTrainer

    cfg = TrainConfig(total_steps=300, eval_interval=10**9,
                      ckpt_interval=10**9, num_envs=1, seed=42,
                      run_dir=str(tmp_path))
    tr = OffPolicyTrainer(_factory(), _track(), _sac(), cfg)
    env = tr.env
    calls_after_done = [0]
    was_done = [False]
    orig_step = env.step

    def spy(action):
        if was_done[0]:
            calls_after_done[0] += 1
        out = orig_step(action)
        was_done[0] = out[2] or out[3]
        return out

    env.step = spy
    tr.train()
    assert calls_after_done[0] == 0


def test_resume_continues_counters(tmp_path):
    from agentRL.core.config import TrainConfig
    from agentRL.train.trainer import OffPolicyTrainer
    from agentRL.checkpoints.io import read_run_state

    d1, d2 = tmp_path / "r1", tmp_path / "r2"
    cfg1 = TrainConfig(total_steps=150, eval_interval=10**9,
                       ckpt_interval=150, num_envs=1, seed=42,
                       run_dir=str(d1))
    tr1 = OffPolicyTrainer(_factory(), _track(), _sac(seed=1), cfg1)
    tr1.train()
    st = read_run_state(str(d1))
    assert st["timestep"] == 150

    cfg2 = TrainConfig(total_steps=300, eval_interval=10**9,
                       ckpt_interval=300, num_envs=1, seed=42,
                       run_dir=str(d2),
                       resume_from=str(d1 / "checkpoints" / "latest.pt"))
    tr2 = OffPolicyTrainer(_factory(), _track(), _sac(seed=1), cfg2)
    tr2.train()
    st2 = read_run_state(str(d2))
    assert st2["timestep"] == 300
    assert tr2.agent.train_state["steps"] >= 150


def test_load_agent_dispatch(tmp_path):
    from agentRL.checkpoints.io import load_agent, save_checkpoint
    from agentRL.algos.sac import SACAgent

    agent = _sac(seed=3)
    p = tmp_path / "a.pt"
    save_checkpoint(agent, str(p))
    loaded = load_agent(str(p))
    assert isinstance(loaded, SACAgent)


def test_mixed_trainer_steps_current_env(tmp_path):
    """Regression: MixedTrackTrainer swaps self.env per track sample — the
    training loop must step the CURRENT env each iteration. Pre-fix, the
    captured local went stale and produced step-after-done on every step
    after the first track switch (E004: 79,593 failure events)."""
    from agentRL.core.config import TrainConfig
    from agentRL.train.trainer import MixedTrackTrainer
    from agentRL.envs.track_registry import TrackRegistry

    reg = TrackRegistry.default()
    tracks = [reg.load("smoke"), reg.load("oval")]
    cfg = TrainConfig(total_steps=400, eval_interval=10**9,
                      ckpt_interval=10**9, num_envs=1, seed=42,
                      run_dir=str(tmp_path))
    tr = MixedTrackTrainer(_factory(), tracks, _sac(), cfg)
    # force alternating tracks so a swap definitely happens
    keys = list(tr.sampler.envs.keys())
    calls = {"n": 0}
    def alt():
        calls["n"] += 1
        return tr.sampler.envs[keys[calls["n"] % 2]]
    tr.sampler.next = alt
    tr.train()
    rows = [json.loads(l) for l in
            open(tmp_path / "metrics.jsonl", encoding="utf-8")]
    failures = [r for r in rows if r["scope"] == "failure"]
    assert not failures, f"{len(failures)} step-after-done failures"
