"""
Domain Randomization Designer.
Supports parametric randomization with Fixed, Uniform, and Normal distributions,
and deterministic seed derivation for reproducible RL experiments.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class DistributionType:
    FIXED = "fixed"
    UNIFORM = "uniform"
    NORMAL = "normal"


@dataclass
class RandomParamConfig:
    """
    Configuration for a single randomized parameter.
    For FIXED: uses param1 as constant value.
    For UNIFORM: uses [param1, param2] as [low, high].
    For NORMAL: uses param1 as mean, param2 as std_dev.
    """
    param_name: str
    distribution: str = DistributionType.FIXED
    param1: float = 1.0       # fixed_val or uniform_min or normal_mean
    param2: float = 0.0       # unused or uniform_max or normal_std
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def sample(self, rng: np.random.Generator) -> float:
        if self.distribution == DistributionType.FIXED:
            val = self.param1
        elif self.distribution == DistributionType.UNIFORM:
            val = float(rng.uniform(self.param1, self.param2))
        elif self.distribution == DistributionType.NORMAL:
            val = float(rng.normal(self.param1, max(1e-6, self.param2)))
        else:
            val = self.param1

        if self.clip_min is not None:
            val = max(self.clip_min, val)
        if self.clip_max is not None:
            val = min(self.clip_max, val)
        return float(val)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RandomParamConfig:
        return cls(
            param_name=str(data.get("param_name", "param")),
            distribution=str(data.get("distribution", DistributionType.FIXED)),
            param1=float(data.get("param1", 1.0)),
            param2=float(data.get("param2", 0.0)),
            clip_min=float(data["clip_min"]) if data.get("clip_min") is not None else None,
            clip_max=float(data["clip_max"]) if data.get("clip_max") is not None else None,
        )


@dataclass
class DomainRandomizationDefinition:
    """
    Collection of authorable randomized simulation parameters.
    """
    enabled: bool = False
    global_seed: int = 42
    parameters: Dict[str, RandomParamConfig] = field(default_factory=dict)

    @classmethod
    def create_default(cls) -> DomainRandomizationDefinition:
        params = {
            "vehicle_mass_mult": RandomParamConfig(
                param_name="vehicle_mass_mult",
                distribution=DistributionType.UNIFORM,
                param1=0.85,
                param2=1.15,
                clip_min=0.5,
                clip_max=2.0
            ),
            "tire_friction_mult": RandomParamConfig(
                param_name="tire_friction_mult",
                distribution=DistributionType.UNIFORM,
                param1=0.80,
                param2=1.20,
                clip_min=0.4,
                clip_max=2.0
            ),
            "surface_friction_mult": RandomParamConfig(
                param_name="surface_friction_mult",
                distribution=DistributionType.UNIFORM,
                param1=0.75,
                param2=1.10,
                clip_min=0.3,
                clip_max=2.0
            ),
            "sensor_noise_mult": RandomParamConfig(
                param_name="sensor_noise_mult",
                distribution=DistributionType.UNIFORM,
                param1=0.5,
                param2=2.0,
                clip_min=0.0,
                clip_max=5.0
            ),
            "spawn_lateral_jitter_m": RandomParamConfig(
                param_name="spawn_lateral_jitter_m",
                distribution=DistributionType.UNIFORM,
                param1=-1.0,
                param2=1.0,
                clip_min=-3.0,
                clip_max=3.0
            ),
            "spawn_heading_jitter_deg": RandomParamConfig(
                param_name="spawn_heading_jitter_deg",
                distribution=DistributionType.NORMAL,
                param1=0.0,
                param2=5.0,
                clip_min=-30.0,
                clip_max=30.0
            ),
        }
        return cls(enabled=False, global_seed=42, parameters=params)

    def sample_all(self, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """
        Samples all randomized parameters deterministically.
        If rng is None, creates a fresh generator from global_seed.
        """
        if not self.enabled:
            return {
                "vehicle_mass_mult": 1.0,
                "tire_friction_mult": 1.0,
                "surface_friction_mult": 1.0,
                "sensor_noise_mult": 1.0,
                "spawn_lateral_jitter_m": 0.0,
                "spawn_heading_jitter_deg": 0.0,
            }

        generator = rng if rng is not None else np.random.default_rng(self.global_seed)
        samples = {}
        for name, param_cfg in self.parameters.items():
            samples[name] = param_cfg.sample(generator)
        return samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "global_seed": self.global_seed,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainRandomizationDefinition:
        raw_params = data.get("parameters", {})
        params = {k: RandomParamConfig.from_dict(v) for k, v in raw_params.items()}
        return cls(
            enabled=bool(data.get("enabled", False)),
            global_seed=int(data.get("global_seed", 42)),
            parameters=params
        )
