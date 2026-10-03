"""TCP end-to-end test — headless sim process + SimGymEnv round-trip.

The in-process path is the training fast lane; this test proves the
network contract still works (protocol 2.1, SET_SCENARIO, 5-tuple).
Marked slow — spawns a real simulator process.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_env_tcp_roundtrip():
    from sim_client import SimulationClient, SimGymEnv
    from sim_experiment.headless import HeadlessSimProcessPool

    pool = HeadlessSimProcessPool(num_envs=1)
    try:
        ports = pool.start(timeout_s=45.0)
    except Exception as e:  # pragma: no cover - env-dependent
        pool.stop()
        pytest.skip(f"headless sim did not start: {e}")

    try:
        port = ports[0]
        env = SimGymEnv(port=port)
        obs, info = env.reset()
        assert obs is not None
        rng = np.random.default_rng(0)
        low, high = env.action_space.low, env.action_space.high
        for _ in range(50):
            a = rng.uniform(low, high).astype(np.float32)
            out = env.step(a)
            assert len(out) == 5, "contract: (obs, rew, term, trunc, info)"
            obs, r, term, trunc, info = out
            assert np.isfinite(r)
            if term or trunc:
                obs, info = env.reset()
        assert "reward_breakdown" in info or "termination_reason" in info
        env.close()

        # SET_SCENARIO over raw client (protocol 2.1) — reset=True applies
        # atomically; the server rejects mid-episode updates without it
        client = SimulationClient(port=port)
        client.connect()
        obs, ack = client.set_scenario({
            "scenario_id": "tcp_e2e_obstacle",
            "name": "TCP E2E obstacle",
            "obstacle_overrides": [{
                "name": "e2e cone", "entity_type": "cone",
                "pos": [70.0, 0.0, 0.0], "yaw": 0.0}],
        }, reset=True)
        assert obs is not None
        client.reset()
        state = client.get_state()
        assert state is not None
        client.close()
    finally:
        pool.stop()
