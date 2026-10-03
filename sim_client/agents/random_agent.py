"""
Baseline random exploration agent for testing the environment lifecycle.
"""

from __future__ import annotations
import time
import numpy as np
from sim_client.client import SimulationClient


def run_random_agent(host: str = "127.0.0.1", port: int = 8765, max_steps: int = 300):
    client = SimulationClient(host=host, port=port)
    spec = client.connect()
    print(f"[Random Agent] Connected to {spec['track_name']}.")

    obs, info = client.reset()
    act_type = spec['action_space']['type']

    for step in range(max_steps):
        if act_type == 'discrete':
            num_actions = len(spec['action_space']['discrete_actions'])
            action = int(np.random.randint(0, num_actions))
        else:
            steer = float(np.random.uniform(-0.5, 0.5))
            throttle = float(np.random.uniform(0.2, 0.8))
            brake = 0.0
            action = [steer, throttle, brake]

        obs, reward, terminated, truncated, info = client.step(action)
        if terminated or truncated:
            obs, info = client.reset()

    client.close()
    print("[Random Agent] Done.")


if __name__ == "__main__":
    run_random_agent()
