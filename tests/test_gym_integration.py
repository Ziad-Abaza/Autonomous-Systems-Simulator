"""
Automated test of Gymnasium environment wrapper.
"""

import threading
import time
import pytest
from sim_env import SimulationEnvironment
from sim_net import SimulationServer
from sim_client import SimGymEnv


def test_gymnasium_integration():
    env = SimulationEnvironment()
    server = SimulationServer(env, host="127.0.0.1", port=9877)
    assert server.start() is True

    stop_event = threading.Event()

    def server_worker():
        while not stop_event.is_set():
            server.poll_and_process()
            time.sleep(0.005)

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    time.sleep(0.1)

    try:
        gym_env = SimGymEnv(host="127.0.0.1", port=9877)
        assert hasattr(gym_env, 'action_space')
        assert hasattr(gym_env, 'observation_space')

        obs, info = gym_env.reset(seed=42)
        assert obs.shape == gym_env.observation_space.shape

        for _ in range(5):
            action = gym_env.action_space.sample()
            obs, reward, term, trunc, info = gym_env.step(action)
            assert obs.shape == gym_env.observation_space.shape
            assert isinstance(reward, float)

        gym_env.close()
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        server.stop()
