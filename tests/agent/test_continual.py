"""ContinualTrainer tests — report schema, forgetting math, resume."""
from __future__ import annotations

import json

import numpy as np
import pytest


def _sac(seed=0):
    from agentRL.algos.sac import SACAgent
    from agentRL.core.config import AgentConfig
    from agentRL.obs.spec import ObservationSpec, PRESETS

    spec = ObservationSpec(channel_names=PRESETS["state8"])
    return SACAgent(
        obs_spec=spec,
        act_space={"low": np.array([-1.0, 0.0, 0.0]),
                   "high": np.array([1.0, 1.0, 1.0])},
        cfg=AgentConfig(algo_id="sac", hidden_sizes=(64, 64), lr=3e-4,
                        gamma=0.99, extra={"seed": seed, "warmup": 32,
                                           "batch": 16,
                                           "random_steps": 16}))


def _factory():
    from agentRL.envs.factory import EnvFactory
    from agentRL.obs.spec import ObservationSpec, PRESETS
    return EnvFactory(obs_spec=ObservationSpec(channel_names=PRESETS["state8"]))


def _cfg(run_dir, steps=80, resume=None):
    from agentRL.core.config import TrainConfig
    return TrainConfig(total_steps=steps, eval_interval=10**9,
                       ckpt_interval=10**9, num_envs=1, seed=42,
                       run_dir=str(run_dir), eval_episodes=1,
                       resume_from=resume)


def test_report_schema(tmp_path):
    from agentRL.train.continual import ContinualTrainer, Phase

    ct = ContinualTrainer(_factory(), _sac(), _cfg(tmp_path / "run"),
                          holdout_tracks=["serpentine"], eval_seeds=(1,),
                          eval_max_steps=60)
    report = ct.run([Phase("smoke", steps=80),
                     Phase("oval", steps=80)])
    path = tmp_path / "run" / "continual_report.json"
    assert path.exists()
    assert report["phases"][0]["track_id"] == "smoke"
    # every seen track + holdout gets a cell after each phase
    for phase_key in report["cells"]:
        assert "smoke" in report["cells"][phase_key]
        assert "oval" in report["cells"][phase_key]
        assert "serpentine" in report["cells"][phase_key]
    assert "forgetting" in report["derived"]
    assert "retention_final" in report["derived"]
    # rehearsal buffer rotated through both tracks
    assert report["rehearsal"]["track_order"] == ["smoke", "oval"]


def test_forgetting_math():
    from agentRL.train.continual import _derive_metrics

    cells = {
        "phase_0": {"A": {"mean_return": 10.0},
                    "B": {"mean_return": 1.0}},
        "phase_1": {"A": {"mean_return": 4.0},
                    "B": {"mean_return": 8.0}},
    }
    d = _derive_metrics(cells, trained_on={"A": 0, "B": 1})
    assert d["forgetting"]["A"] == pytest.approx(4.0 - 10.0)
    assert d["forgetting"]["B"] == pytest.approx(8.0 - 8.0)
    assert d["retention_final"]["A"] == 4.0
    assert d["transfer"]["B"] == pytest.approx(8.0 - 1.0)


def test_resume_mid_sequence(tmp_path):
    from agentRL.train.continual import ContinualTrainer, Phase

    phases = [Phase("smoke", steps=60), Phase("oval", steps=60)]
    ct1 = ContinualTrainer(_factory(), _sac(seed=5),
                           _cfg(tmp_path / "seq"), holdout_tracks=[],
                           eval_seeds=(1,), eval_max_steps=40)
    ct1.run(phases[:1])
    ckpt = tmp_path / "seq" / "checkpoints" / "phase_0_smoke.pt"
    assert ckpt.exists()

    ct2 = ContinualTrainer(_factory(), _sac(seed=5),
                           _cfg(tmp_path / "seq", resume=str(ckpt)),
                           holdout_tracks=[], eval_seeds=(1,),
                           eval_max_steps=40)
    report = ct2.run(phases, resume=True)
    assert len(report["cells"]) == 2, "resume should add phase_1 cells"
    assert ct2.agent.train_state["steps"] >= 60
