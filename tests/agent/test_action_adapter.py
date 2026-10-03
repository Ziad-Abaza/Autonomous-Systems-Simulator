"""ActionAdapter tests — normalized [-1,1] agent space <-> env bounds."""
from __future__ import annotations

import numpy as np

from agentRL.act.adapter import ActionAdapter


def _adapter() -> ActionAdapter:
    # env contract: steer [-1,1], throttle [0,1], brake [0,1]
    return ActionAdapter(low=np.array([-1.0, 0.0, 0.0]),
                         high=np.array([1.0, 1.0, 1.0]))


def test_roundtrip():
    a = _adapter()
    raw = np.array([-0.7, 0.3, 0.9], dtype=np.float32)
    env = a.to_env(raw)
    back = a.from_env(env)
    assert np.allclose(back, raw, atol=1e-6)


def test_bounds_mapping():
    a = _adapter()
    env = a.to_env(np.array([-1.0, -1.0, -1.0]))
    assert np.allclose(env, [-1.0, 0.0, 0.0])
    env = a.to_env(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(env, [1.0, 1.0, 1.0])
    env = a.to_env(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(env, [0.0, 0.5, 0.5])


def test_out_of_range_clipped():
    a = _adapter()
    env, valid = a.to_env(np.array([2.5, -3.0, 0.5]), validate=True)
    assert valid
    assert np.allclose(env, [1.0, 0.0, 0.75])


def test_nan_inf_handling():
    a = _adapter()
    env, valid = a.to_env(np.array([np.nan, 0.5, 0.0]), validate=True)
    assert not valid
    assert np.allclose(env, [0.0, 0.75, 0.5])
    env, valid = a.to_env(np.array([np.inf, 0.0, 0.0]), validate=True)
    assert not valid
    assert np.isfinite(env).all()


def test_pos_only_brake():
    """Brake channel is one-sided: raw<0 = brakes released (brake>=0.1
    parks the car in this sim — affine mapping causes permanent stalls)."""
    a = ActionAdapter(low=np.array([-1.0, 0.0, 0.0]),
                      high=np.array([1.0, 1.0, 1.0]), pos_only=(2,))
    env = a.to_env(np.array([0.0, 0.0, -1.0]))
    assert env[2] == 0.0, "negative raw must release brakes"
    env = a.to_env(np.array([0.0, 0.0, 0.4]))
    assert abs(env[2] - 0.4) < 1e-6
    back = a.from_env(env)
    assert abs(back[2] - 0.4) < 1e-6


def test_prev_action_tracking():
    a = _adapter()
    assert a.prev_action is None
    a.to_env(np.array([0.1, 0.2, 0.0]))
    assert a.prev_action is not None
    assert np.allclose(a.prev_action, [0.1, 0.6, 0.5])
    a.reset()
    assert a.prev_action is None
