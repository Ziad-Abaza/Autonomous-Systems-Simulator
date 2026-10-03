"""
Gymnasium-compatible environment wrapper for external RL frameworks.
Supports Stable-Baselines3, CleanRL, PyTorch, Ray RLlib, etc.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from sim_client.client import SimulationClient


class SimGymEnv(gym.Env):
    """
    Gymnasium Environment that connects to the standalone 3D Simulator over TCP.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        super().__init__()
        self.host = host
        self.port = port
        self.client = SimulationClient(host=host, port=port)

        # Connect and get spec
        spec = self.client.connect()
        act_cfg = spec['action_space']
        obs_dim = spec['vector_dim']

        # Configure action space
        if act_cfg['type'] == 'discrete':
            num_actions = len(act_cfg['discrete_actions'])
            self.action_space = spaces.Discrete(num_actions)
        else:
            low = np.array(act_cfg['continuous_low'], dtype=np.float32)
            high = np.array(act_cfg['continuous_high'], dtype=np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Configure observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.client.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: Union[np.ndarray, int, float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.client.step(action)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.client.close()
