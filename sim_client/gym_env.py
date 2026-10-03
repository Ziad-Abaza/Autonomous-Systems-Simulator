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
        obs_schema = spec.get('observation_schema', {})
        include_cam = obs_schema.get('include_camera_rgb', False)
        flatten_vec = obs_schema.get('flatten_vector', True)

        if flatten_vec:
            if include_cam and obs_dim > 0:
                self.observation_space = spaces.Dict({
                    'vector': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                    'image': spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
                })
            elif include_cam and obs_dim == 0:
                self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
            else:
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32
                )
        else:
            space_dict = {}
            if obs_schema.get('include_speed', True):
                space_dict['speed'] = spaces.Box(low=0.0, high=100.0, shape=(), dtype=np.float32)
            if obs_schema.get('include_velocity', True):
                space_dict['vel_body'] = spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)
            if obs_schema.get('include_yaw_rate', True):
                space_dict['yaw_rate'] = spaces.Box(low=-10.0, high=10.0, shape=(), dtype=np.float32)
            if obs_schema.get('include_steering_angle', True):
                space_dict['steering_angle'] = spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
            if obs_schema.get('include_distance_from_center', True):
                space_dict['distance_from_center'] = spaces.Box(low=-50.0, high=50.0, shape=(), dtype=np.float32)
            if obs_schema.get('include_heading_error', True):
                space_dict['heading_error'] = spaces.Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32)
            if obs_schema.get('include_distance_to_checkpoint', True):
                space_dict['distance_to_checkpoint'] = spaces.Box(low=0.0, high=1000.0, shape=(), dtype=np.float32)
            if obs_schema.get('include_lidar_rays', True):
                space_dict['lidar_ranges'] = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
            if include_cam:
                space_dict['camera_rgb'] = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
            self.observation_space = spaces.Dict(space_dict)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self.client.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: Union[np.ndarray, int, float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.client.step(action)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.client.close()
