"""Eval suite tests — evaluate_policy, failure classifier, eval matrix."""
from __future__ import annotations

import json

import numpy as np
import pytest


def _factory():
    from agentRL.envs.factory import EnvFactory
    from agentRL.obs.spec import ObservationSpec, PRESETS
    return EnvFactory(obs_spec=ObservationSpec(channel_names=PRESETS["state8"]))


def _track(tid="smoke"):
    from agentRL.envs.track_registry import TrackRegistry
    return TrackRegistry.default().load(tid)


class _Random:
    algo_id = "random"
    obs_spec = None

    def act(self, obs, deterministic=False):
        return np.array([np.random.uniform(-1, 1),
                         np.random.uniform(0, 1), 0.0]), {}


class _Still:
    algo_id = "stationary"

    def act(self, obs, deterministic=False):
        return np.array([0.0, 0.0, 0.0]), {}


def test_evaluate_returns_aggregate():
    from agentRL.eval.evaluate import evaluate_policy

    out = evaluate_policy(_factory(), _track(), _Random(),
                          seeds=(1, 2), max_steps=120)
    agg = out["aggregate"]
    for k in ("mean_return", "completion_rate", "collision_rate",
              "off_road_rate", "timeout_rate", "mean_progress",
              "mean_speed", "steer_smoothness"):
        assert k in agg, f"missing metric {k}"
    for k in ("completion_rate", "collision_rate", "off_road_rate",
              "timeout_rate"):
        assert 0.0 <= agg[k] <= 1.0
    assert len(out["per_episode"]) == 2


def test_evaluate_stationary_is_classified_stall():
    from agentRL.eval.evaluate import evaluate_policy

    out = evaluate_policy(_factory(), _track(), _Still(),
                          seeds=(1,), max_steps=200)
    ep = out["per_episode"][0]
    assert ep["failure_class"] == "stall_timeout", ep


def test_classify_failure_mapping():
    from agentRL.eval.failures import classify_failure

    def rec(reason, speed=0.0, progress=0.0):
        return {"termination_reason": reason, "mean_speed": speed,
                "lap_progress": progress, "steer_sign_flips": 0.0}

    assert classify_failure(rec("collision")) == "collision"
    assert classify_failure(rec("off_road")) == "off_track"
    assert classify_failure(rec("wrong_direction")) == "wrong_direction"
    assert classify_failure(rec("completion")) == "completed"
    assert classify_failure(rec("max_steps", speed=0.0)) == "stall_timeout"
    assert classify_failure(rec("max_steps", speed=8.0,
                              progress=0.6)) == "timeout_progress"
    assert classify_failure(rec("stuck", speed=0.0)) == "stall_timeout"
    assert classify_failure(rec("stuck", speed=5.0)) == "no_progress"
    assert classify_failure(rec("max_duration_exceeded",
                                speed=0.0)) == "stall_timeout"


def test_classify_oscillation():
    from agentRL.eval.failures import classify_failure

    rec = {"termination_reason": "stuck", "mean_speed": 4.0,
           "lap_progress": 0.1, "steer_sign_flips": 0.6}
    assert classify_failure(rec) == "oscillation"


def test_matrix_grid_shape(tmp_path):
    from agentRL.eval.evaluate import evaluate_policy
    from agentRL.eval.matrix import EvalMatrix

    mx = EvalMatrix(_factory())
    ckpts = {"random": _Random(), "still": _Still()}
    out = mx.run(ckpts, [_track("smoke"), _track("oval")],
                 out_dir=str(tmp_path), seeds=(1,), max_steps=80)
    assert set(out.keys()) == {"random", "still"}
    for cells in out.values():
        assert set(cells.keys()) == {"smoke", "oval"}
    assert (tmp_path / "eval_matrix.json").exists()
