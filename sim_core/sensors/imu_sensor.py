"""
Inertial Measurement Unit (IMU) sensor simulating 3-axis accelerometer and 3-axis gyroscope.
Includes bias drift and Gaussian measurement noise.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from sim_core.math_utils import Vec3
from sim_core.sensors.base_sensor import BaseSensor


class IMUSensor(BaseSensor):
    """
    Simulates a vehicle-mounted 6-axis IMU.
    """
    def __init__(
        self,
        name: str = "imu",
        update_frequency_hz: float = 100.0,
        accel_noise_std: float = 0.05,
        gyro_noise_std: float = 0.01,
        bias_drift_rate: float = 0.001,
        local_pos: Optional[Vec3] = None
    ):
        super().__init__(
            name=name,
            sensor_type="imu",
            update_frequency_hz=update_frequency_hz,
            local_pos=local_pos or Vec3(0.0, 0.0, 0.3)
        )
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.bias_drift_rate = bias_drift_rate

        self.accel_bias = np.zeros(3, dtype=np.float32)
        self.gyro_bias = np.zeros(3, dtype=np.float32)

    def _generate_raw_sample(self, context: Any, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        vehicle = context['vehicle']
        st = vehicle.state

        # Accelerometer body axes: ax, ay, az (including gravity +9.81 on Z)
        ax = float(st.accel_body.x)
        ay = float(st.accel_body.y)
        az = 9.81

        # Gyroscope rates: roll_rate, pitch_rate, yaw_rate
        p_rate = 0.0
        q_rate = 0.0
        r_rate = float(st.yaw_rate)

        # Drift biases slightly
        self.accel_bias += rng.normal(0.0, self.bias_drift_rate, size=3).astype(np.float32)
        self.gyro_bias += rng.normal(0.0, self.bias_drift_rate, size=3).astype(np.float32)

        # Noise
        accel_noise = rng.normal(0.0, self.accel_noise_std, size=3).astype(np.float32)
        gyro_noise = rng.normal(0.0, self.gyro_noise_std, size=3).astype(np.float32)

        accel = np.array([ax, ay, az], dtype=np.float32) + self.accel_bias + accel_noise
        gyro = np.array([p_rate, q_rate, r_rate], dtype=np.float32) + self.gyro_bias + gyro_noise

        return {
            'linear_acceleration': accel,
            'angular_velocity': gyro,
        }

    def reset(self) -> None:
        super().reset()
        self.accel_bias.fill(0.0)
        self.gyro_bias.fill(0.0)
