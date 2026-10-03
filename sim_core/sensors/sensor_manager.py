"""
Sensor Manager orchestrating modular sensor instances, scheduling updates,
and aggregating samples.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
from sim_core.sensors.base_sensor import BaseSensor
from sim_core.sensors.vehicle_state_sensor import VehicleStateSensor
from sim_core.sensors.raycast_sensor import RaycastSensor
from sim_core.sensors.imu_sensor import IMUSensor
from sim_core.sensors.camera_sensor import CameraSensor


class SensorManager:
    """
    Manages all attached sensors on an agent/vehicle.
    """
    def __init__(self):
        self.sensors: Dict[str, BaseSensor] = {}

    def add_sensor(self, sensor: BaseSensor) -> None:
        self.sensors[sensor.name] = sensor

    def remove_sensor(self, name: str) -> None:
        if name in self.sensors:
            del self.sensors[name]

    def get_sensor(self, name: str) -> Optional[BaseSensor]:
        return self.sensors.get(name)

    def reset_all(self) -> None:
        for sensor in self.sensors.values():
            sensor.reset()

    def update_all(self, sim_time: float, context: Any, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Updates each sensor according to its internal frequency and returns dict of samples.
        """
        results = {}
        for name, sensor in self.sensors.items():
            results[name] = sensor.update(sim_time, context, rng)
        return results

    def get_all_samples(self) -> Dict[str, Any]:
        return {name: s.get_last_sample() for name, s in self.sensors.items()}

    @classmethod
    def create_default_sensor_suite(cls) -> SensorManager:
        """Standard sensor suite: State + 15-beam LiDAR + RGB camera + IMU."""
        sm = cls()
        sm.add_sensor(VehicleStateSensor(name="vehicle_state", update_frequency_hz=60.0))
        sm.add_sensor(RaycastSensor(name="lidar_rays", num_rays=15, fov_degrees=180.0, max_range=40.0, update_frequency_hz=30.0))
        sm.add_sensor(CameraSensor(name="rgb_camera", width=84, height=84, fov_degrees=75.0, update_frequency_hz=30.0))
        sm.add_sensor(IMUSensor(name="imu", update_frequency_hz=60.0))
        return sm
