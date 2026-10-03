"""
End-to-end integration test of TCP SimulationServer and SimulationClient.
"""

import threading
import time
import pytest
from sim_env import SimulationEnvironment
from sim_net import SimulationServer
from sim_client import SimulationClient


def test_server_client_handshake_and_step():
    env = SimulationEnvironment()
    server = SimulationServer(env, host="127.0.0.1", port=9876)
    assert server.start() is True

    # Server worker thread
    stop_event = threading.Event()

    def server_worker():
        while not stop_event.is_set():
            server.poll_and_process()
            time.sleep(0.005)

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    time.sleep(0.1)

    try:
        # Client connects
        client = SimulationClient(host="127.0.0.1", port=9876)
        spec = client.connect()
        assert spec['track_name'] == env.road_def.name
        assert 'action_space' in spec
        assert spec['physics_hz'] == 60.0

        # Client resets
        obs, info = client.reset(seed=123)
        assert len(obs) > 0
        assert info['termination_reason'] == "reset"

        # Client steps
        for _ in range(10):
            obs, reward, term, trunc, info = client.step([0.0, 0.6, 0.0])
            assert not term
            assert 'reward_breakdown' in info
            assert info['speed'] >= 0.0

        client.close()
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        server.stop()
