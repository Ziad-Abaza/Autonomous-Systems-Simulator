"""
Domain Randomization engine for RL generalization.
Supports seedable randomization of physical properties, sensor noise,
surface friction, and spawn poses.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class DomainRandomizationConfig:
    enabled: bool = False
    # Multipliers on baseline properties [min, max]
    mass_range: Tuple[float, float] = (0.85, 1.15)
    tire_friction_range: Tuple[float, float] = (0.80, 1.20)
    surface_friction_range: Tuple[float, float] = (0.75, 1.10)
    sensor_noise_multiplier_range: Tuple[float, float] = (0.5, 2.0)
    # Spawn pose variations
    spawn_lateral_jitter_m: float = 1.0     # lateral jitter across road width
    spawn_heading_jitter_deg: float = 10.0  # yaw angle jitter

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'mass_range': list(self.mass_range),
            'tire_friction_range': list(self.tire_friction_range),
            'surface_friction_range': list(self.surface_friction_range),
            'sensor_noise_multiplier_range': list(self.sensor_noise_multiplier_range),
            'spawn_lateral_jitter_m': self.spawn_lateral_jitter_m,
            'spawn_heading_jitter_deg': self.spawn_heading_jitter_deg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainRandomizationConfig:
        cfg = cls()
        cfg.enabled = bool(data.get('enabled', False))
        if 'mass_range' in data:
            cfg.mass_range = tuple(data['mass_range'])
        if 'tire_friction_range' in data:
            cfg.tire_friction_range = tuple(data['tire_friction_range'])
        if 'surface_friction_range' in data:
            cfg.surface_friction_range = tuple(data['surface_friction_range'])
        if 'sensor_noise_multiplier_range' in data:
            cfg.sensor_noise_multiplier_range = tuple(data['sensor_noise_multiplier_range'])
        cfg.spawn_lateral_jitter_m = float(data.get('spawn_lateral_jitter_m', 1.0))
        cfg.spawn_heading_jitter_deg = float(data.get('spawn_heading_jitter_deg', 10.0))
        return cfg


class DomainRandomizer:
    """
    Applies bounded domain randomization to simulation components.
    """
    def __init__(self, config: DomainRandomizationConfig | None = None):
        self.config = config or DomainRandomizationConfig()

    def sample_parameters(self, rng: np.random.Generator) -> Dict[str, float]:
        cfg = self.config
        if not cfg.enabled:
            return {
                'mass_factor': 1.0,
                'tire_friction_factor': 1.0,
                'surface_friction_factor': 1.0,
                'sensor_noise_factor': 1.0,
                'spawn_lateral_jitter': 0.0,
                'spawn_heading_jitter': 0.0,
            }

        return {
            'mass_factor': float(rng.uniform(*cfg.mass_range)),
            'tire_friction_factor': float(rng.uniform(*cfg.tire_friction_range)),
            'surface_friction_factor': float(rng.uniform(*cfg.surface_friction_range)),
            'sensor_noise_factor': float(rng.uniform(*cfg.sensor_noise_multiplier_range)),
            'spawn_lateral_jitter': float(rng.uniform(-cfg.spawn_lateral_jitter_m, cfg.spawn_lateral_jitter_m)),
            'spawn_heading_jitter': float(np.radians(rng.uniform(-cfg.spawn_heading_jitter_deg, cfg.spawn_heading_jitter_deg))),
        }
