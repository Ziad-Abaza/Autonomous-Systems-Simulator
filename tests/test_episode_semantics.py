"""Episode lifecycle semantics — the documented step-after-done contract.

Contract (docs/rl-overview.md, docs/troubleshooting.md): a real terminal
step always has exactly one of {terminated, truncated} set. A step() called
while is_done is a non-step: it returns the sentinel
(obs, 0.0, True, True, info) with info['error'] and
termination_reason='invalid_call_after_done'. The both-flags combination
is intentionally reserved to mean "invalid call" — it can never be produced
by a real terminal step, so it is unambiguous for clients.

These tests prove the contract end-to-end, incl. over TCP.
"""
import threading
import time
import numpy as np
import pytest

from sim_env import SimulationEnvironment


def _force_terminated(env):
    env.is_done = True
    env.last_termination_reason_dict = {
        "reason": "collision", "step": env.current_step,
        "sim_time": env.clock.sim_time, "is_truncation": False}


def _force_truncated(env):
    env.is_done = True
    env.last_termination_reason_dict = {
        "reason": "checkpoint_timeout", "step": env.current_step,
        "sim_time": env.clock.sim_time, "is_truncation": True}


def test_real_terminal_step_has_single_flag():
    """A real collision ends with terminated=True, truncated=False."""
    env = SimulationEnvironment()
    env.reset(seed=1)
    # drive straight into a barrier/obstacle at full throttle
    term = trunc = False
    for _ in range(1800):
        _, _, term, trunc, _ = env.step([0.0, 1.0, 0.0])
        if term or trunc:
            break
    assert term != trunc, "a real terminal step must carry exactly one flag"


def test_step_after_terminated_is_sentinel():
    env = SimulationEnvironment()
    env.reset(seed=1)
    _force_terminated(env)
    obs, r, t, tr, info = env.step([0.0, 1.0, 0.0])
    assert (t, tr) == (True, True)
    assert r == 0.0
    assert info["termination_reason"] == "invalid_call_after_done"
    assert "error" in info


def test_step_after_truncated_is_sentinel():
    env = SimulationEnvironment()
    env.reset(seed=1)
    _force_truncated(env)
    obs, r, t, tr, info = env.step([0.0, 1.0, 0.0])
    assert (t, tr) == (True, True)
    assert r == 0.0
    assert info["termination_reason"] == "invalid_call_after_done"


def test_repeated_post_done_steps_are_stable_sentinels():
    env = SimulationEnvironment()
    env.reset(seed=1)
    _force_terminated(env)
    for _ in range(3):
        _, r, t, tr, info = env.step([0.0, 0.5, 0.0])
        assert (r, t, tr) == (0.0, True, True)
        assert info["termination_reason"] == "invalid_call_after_done"


def test_reset_restores_stepping():
    env = SimulationEnvironment()
    env.reset(seed=1)
    _force_terminated(env)
    env.step([0.0, 0.5, 0.0])  # sentinel
    env.reset(seed=1)
    _, r, t, tr, info = env.step([0.0, 0.5, 0.0])
    assert not (t and tr)
    assert info["termination_reason"] != "invalid_call_after_done"


def test_post_done_step_does_not_advance_episode():
    env = SimulationEnvironment()
    env.reset(seed=1)
    for _ in range(5):
        env.step([0.0, 0.5, 0.0])
    step_before = env.current_step
    time_before = env.clock.sim_time
    _force_terminated(env)
    env.step([0.0, 0.5, 0.0])
    assert env.current_step == step_before
    assert env.clock.sim_time == time_before


def test_tcp_step_after_done_same_contract():
    from sim_net import SimulationServer
    from sim_client import SimulationClient
    env = SimulationEnvironment()
    server = SimulationServer(env, host="127.0.0.1", port=9895)
    assert server.start() is True
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            server.poll_and_process()
            time.sleep(0.003)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.1)
    try:
        client = SimulationClient(host="127.0.0.1", port=9895)
        client.connect()
        client.reset()
        _force_terminated(env)
        obs, r, term, trunc, info = client.step([0.0, 0.5, 0.0])
        assert (term, trunc) == (True, True)
        assert r == 0.0
        assert info["termination_reason"] == "invalid_call_after_done"
    finally:
        stop.set()
