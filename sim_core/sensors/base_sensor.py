"""
Modular sensor base architecture with independent update frequencies,
noise models, and latency simulation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from sim_core.math_utils import Vec3


class BaseSensor(ABC):
    """
    Abstract base class for all simulation sensors.
    """
    def __init__(
        self,
        name: str,
        sensor_type: str,
        update_frequency_hz: float = 30.0,
        local_pos: Optional[Vec3] = None,
        local_yaw: float = 0.0,
        noise_std: float = 0.0,
        latency_seconds: float = 0.0
    ):
        self.name = name
        self.sensor_type = sensor_type
        self.update_frequency_hz = max(1.0, float(update_frequency_hz))
        self.update_interval = 1.0 / self.update_frequency_hz
        self.local_pos = local_pos or Vec3(0.0, 0.0, 1.2)  # Relative to vehicle
        self.local_yaw = local_yaw                         # Relative to vehicle heading
        self.noise_std = float(noise_std)
        self.latency_seconds = float(latency_seconds)

        # Timing and state
        self._last_update_time: float = -999.0
        self._last_sample: Any = None
        self._history_buffer: list[tuple[float, Any]] = []

    def should_update(self, current_sim_time: float) -> bool:
        """Determines if the sensor needs to produce a new measurement."""
        return (current_sim_time - self._last_update_time) >= (self.update_interval - 1e-6)

    def update(self, current_sim_time: float, context: Any, rng: np.random.Generator) -> Any:
        """
        Updates sensor measurement if its sampling interval has elapsed.
        Applies latency buffer and noise.
        """
        if self.should_update(current_sim_time) or self._last_sample is None:
            raw_data = self._generate_raw_sample(context, rng)
            noisy_data = self._apply_noise(raw_data, rng)
            self._last_update_time = current_sim_time

            if self.latency_seconds > 0.0:
                self._history_buffer.append((current_sim_time, noisy_data))
                # Purge old samples
                cutoff = current_sim_time - self.latency_seconds * 2.0
                self._history_buffer = [item for item in self._history_buffer if item[0] >= cutoff]

                # Find sample closest to current_sim_time - latency_seconds
                target_time = current_sim_time - self.latency_seconds
                for t, data in reversed(self._history_buffer):
                    if t <= target_time:
                        self._last_sample = data
                        break
                else:
                    self._last_sample = noisy_data
            else:
                self._last_sample = noisy_data

        return self._last_sample

    def get_last_sample(self) -> Any:
        return self._last_sample

    def reset(self) -> None:
        self._last_update_time = -999.0
        self._last_sample = None
        self._history_buffer.clear()

    @abstractmethod
    def _generate_raw_sample(self, context: Any, rng: np.random.Generator) -> Any:
        """Domain-specific raw measurement generation."""
        pass

    def _apply_noise(self, data: Any, rng: np.random.Generator) -> Any:
        """Default noise application."""
        if self.noise_std <= 0.0:
            return data
        if isinstance(data, np.ndarray):
            noise = rng.normal(0.0, self.noise_std, size=data.shape).astype(data.dtype)
            return data + noise
        elif isinstance(data, (float, int)):
            return data + rng.normal(0.0, self.noise_std)
        return data

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.sensor_type,
            'update_frequency_hz': self.update_frequency_hz,
            'local_pos': self.local_pos.to_tuple(),
            'local_yaw': self.local_yaw,
            'noise_std': self.noise_std,
            'latency_seconds': self.latency_seconds,
        }
