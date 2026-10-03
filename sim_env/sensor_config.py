"""
Declarative sensor-suite configuration.

A ``SensorConfig`` is the *serialized* description of one sensor attached to
an agent — name, type, enable flag, and type-specific parameters. The runtime
``SensorManager`` is rebuilt from these configs so every property the engine
supports is persisted in ``*.sim.json`` and editable in the inspector.

Type registry:
    camera_rgb      RGB camera  — width, height, fov_degrees,
                                  local_pos[3], local_yaw, local_pitch,
                                  update_frequency_hz, noise_std,
                                  latency_seconds
    lidar_rays      planar LiDAR— num_rays, fov_degrees, max_range,
                                  local_pos[3], local_yaw,
                                  update_frequency_hz, noise_std,
                                  latency_seconds
    imu             6-axis IMU  — update_frequency_hz, accel_noise_std,
                                  gyro_noise_std, bias_drift_rate, local_pos[3]
    vehicle_state   kinematics  — update_frequency_hz, noise_std,
                                  latency_seconds
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from sim_core.math_utils import Vec3
from sim_core.sensors.base_sensor import BaseSensor


DEFAULT_CAMERA_PARAMS: Dict[str, Any] = {
    "width": 84, "height": 84, "fov_degrees": 75.0,
    "update_frequency_hz": 30.0,
    "local_pos": [1.0, 0.0, 1.1], "local_yaw": 0.0, "local_pitch": -0.05,
    "noise_std": 0.0, "latency_seconds": 0.0,
}
DEFAULT_LIDAR_PARAMS: Dict[str, Any] = {
    "num_rays": 15, "fov_degrees": 180.0, "max_range": 40.0,
    "update_frequency_hz": 30.0,
    "local_pos": [0.0, 0.0, 1.2], "local_yaw": 0.0,
    "noise_std": 0.0, "latency_seconds": 0.0,
}
DEFAULT_IMU_PARAMS: Dict[str, Any] = {
    "update_frequency_hz": 100.0,
    "accel_noise_std": 0.05, "gyro_noise_std": 0.01,
    "bias_drift_rate": 0.001, "local_pos": [0.0, 0.0, 0.5],
}
DEFAULT_STATE_PARAMS: Dict[str, Any] = {
    "update_frequency_hz": 60.0, "noise_std": 0.0, "latency_seconds": 0.0,
}

_SENSOR_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "camera_rgb": DEFAULT_CAMERA_PARAMS,
    "lidar_rays": DEFAULT_LIDAR_PARAMS,
    "imu": DEFAULT_IMU_PARAMS,
    "vehicle_state": DEFAULT_STATE_PARAMS,
}

_SENSOR_TYPE_NAMES = {
    "camera_rgb": "RGB Camera",
    "lidar_rays": "LiDAR",
    "imu": "IMU",
    "vehicle_state": "Vehicle State",
}

# Legacy name → type inference for older files that only had sensor_names.
_NAME_TYPE_HINTS = {
    "rgb_camera": "camera_rgb",
    "lidar_rays": "lidar_rays",
    "imu": "imu",
    "vehicle_state": "vehicle_state",
}


def infer_sensor_type(name: str) -> str:
    """Best-effort type guess for a legacy sensor name."""
    if name in _NAME_TYPE_HINTS:
        return _NAME_TYPE_HINTS[name]
    lower = name.lower()
    if "cam" in lower:
        return "camera_rgb"
    if "lidar" in lower or "ray" in lower:
        return "lidar_rays"
    if "imu" in lower:
        return "imu"
    return "vehicle_state"


@dataclass
class SensorConfig:
    """Serialized description of one attached sensor."""
    name: str
    sensor_type: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def for_type(cls, sensor_type: str, name: str) -> "SensorConfig":
        return cls(name=name, sensor_type=sensor_type,
                   params=dict(_SENSOR_DEFAULTS.get(sensor_type, {})))

    @property
    def display_type(self) -> str:
        return _SENSOR_TYPE_NAMES.get(self.sensor_type, self.sensor_type)

    @property
    def is_camera(self) -> bool:
        return self.sensor_type == "camera_rgb"

    def merged_params(self) -> Dict[str, Any]:
        """Defaults for the sensor type overlaid with stored overrides."""
        merged = dict(_SENSOR_DEFAULTS.get(self.sensor_type, {}))
        merged.update(self.params or {})
        return merged

    # ------------------------------------------------------------------
    def build_sensor(self) -> Optional[BaseSensor]:
        """Instantiate the concrete runtime sensor for this config."""
        if not self.enabled or self.sensor_type not in _SENSOR_DEFAULTS:
            return None
        p = self.merged_params()
        lp = p.get("local_pos") or [0.0, 0.0, 0.0]
        local_pos = Vec3(float(lp[0]), float(lp[1]), float(lp[2]))
        from sim_core.sensors.camera_sensor import CameraSensor
        from sim_core.sensors.raycast_sensor import RaycastSensor
        from sim_core.sensors.imu_sensor import IMUSensor
        from sim_core.sensors.vehicle_state_sensor import VehicleStateSensor

        if self.sensor_type == "camera_rgb":
            return CameraSensor(
                name=self.name,
                width=int(p["width"]), height=int(p["height"]),
                fov_degrees=float(p["fov_degrees"]),
                update_frequency_hz=float(p["update_frequency_hz"]),
                local_pos=local_pos,
                local_yaw=float(p["local_yaw"]),
                local_pitch=float(p["local_pitch"]),
                noise_std=float(p["noise_std"]),
                latency_seconds=float(p["latency_seconds"]))
        if self.sensor_type == "lidar_rays":
            return RaycastSensor(
                name=self.name,
                num_rays=int(p["num_rays"]),
                fov_degrees=float(p["fov_degrees"]),
                max_range=float(p["max_range"]),
                update_frequency_hz=float(p["update_frequency_hz"]),
                local_pos=local_pos,
                local_yaw=float(p["local_yaw"]),
                noise_std=float(p["noise_std"]),
                latency_seconds=float(p["latency_seconds"]))
        if self.sensor_type == "imu":
            return IMUSensor(
                name=self.name,
                update_frequency_hz=float(p["update_frequency_hz"]),
                accel_noise_std=float(p["accel_noise_std"]),
                gyro_noise_std=float(p["gyro_noise_std"]),
                bias_drift_rate=float(p["bias_drift_rate"]),
                local_pos=local_pos)
        return VehicleStateSensor(
            name=self.name,
            update_frequency_hz=float(p["update_frequency_hz"]),
            noise_std=float(p["noise_std"]),
            latency_seconds=float(p["latency_seconds"]))

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sensor_type": self.sensor_type,
            "enabled": self.enabled,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorConfig":
        return cls(
            name=str(data.get("name", "sensor")),
            sensor_type=str(data.get("sensor_type",
                                     infer_sensor_type(
                                         str(data.get("name", ""))))),
            enabled=bool(data.get("enabled", True)),
            params=dict(data.get("params") or {}),
        )


def default_suite_configs() -> List[SensorConfig]:
    """The four-sensor default suite, mirroring the legacy runtime default."""
    return [
        SensorConfig.for_type("vehicle_state", "vehicle_state"),
        SensorConfig.for_type("lidar_rays", "lidar_rays"),
        SensorConfig.for_type("camera_rgb", "rgb_camera"),
        SensorConfig.for_type("imu", "imu"),
    ]


def configs_from_legacy_names(names: List[str]) -> List[SensorConfig]:
    """Upgrade a legacy ``sensor_names`` list to full configs."""
    return [SensorConfig.for_type(infer_sensor_type(n), n) for n in names]
