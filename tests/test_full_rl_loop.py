"""
Full End-to-End Vertical Slice Test of the RL Loop.
Launches the simulator server, connects an external AI model,
executes closed-loop control steps, and validates observation, reward, and physics telemetry.
"""

import subprocess
import time
import sys
import os
import pytest
import numpy as np

from sim_client import SimulationClient


def test_full_autonomous_driving_loop():
    port = 8790
    cmd = [sys.executable, "-u", "main.py", "--headless", "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    time.sleep(1.5)  # Wait for server to bind

    try:
        client = SimulationClient(host="127.0.0.1", port=port, timeout=5.0)
        spec = client.connect()
        assert spec['track_name'] == "Proving Ground Circuit"
        assert spec['physics_hz'] == 60.0
        assert spec['vector_dim'] == 23

        obs, info = client.reset(seed=100)
        assert len(obs) == 23
        assert info['speed'] == 0.0

        # Step forward with throttle
        total_reward = 0.0
        speeds = []
        for step in range(30):
            action = [0.0, 0.7, 0.0]  # Drive straight with 70% throttle
            obs, reward, term, trunc, info = client.step(action)
            total_reward += reward
            speeds.append(info['speed'])
            assert not term
            assert 'reward_breakdown' in info
            assert 'progress' in info['reward_breakdown']
            assert 'centering' in info['reward_breakdown']

        # Verify physical progression
        assert speeds[-1] > speeds[0], "Vehicle should accelerate under throttle"
        assert total_reward > 0.0, "Positive reward should be earned for progress"
        assert len(obs) == 23

        client.close()
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=2.0)
        except Exception:
            proc.kill()
