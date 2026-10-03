"""
Modular dynamic vehicle physics model with 4-wheel tire slip, aerodynamic drag,
suspension roll/pitch, and low-speed singularity blending.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple
import numpy as np
from sim_core.math_utils import Vec2, Vec3, OBB2D, clamp, normalize_angle
from sim_core.vehicle.vehicle_config import VehicleConfig


class VehicleState:
    """Snapshot of vehicle physical state."""
    __slots__ = (
        'pos', 'yaw', 'pitch', 'roll',
        'vel_body', 'vel_world', 'yaw_rate',
        'steering_angle', 'throttle', 'brake',
        'accel_body', 'accel_world',
        'wheel_angles', 'wheel_speeds',
        'is_colliding'
    )

    def __init__(self):
        self.pos = Vec3(0.0, 0.0, 0.2)
        self.yaw: float = 0.0
        self.pitch: float = 0.0
        self.roll: float = 0.0

        # Body frame velocities: vx (forward), vy (lateral)
        self.vel_body = Vec2(0.0, 0.0)
        self.vel_world = Vec2(0.0, 0.0)
        self.yaw_rate: float = 0.0

        # Control inputs
        self.steering_angle: float = 0.0
        self.throttle: float = 0.0
        self.brake: float = 0.0

        # Accelerations
        self.accel_body = Vec2(0.0, 0.0)
        self.accel_world = Vec2(0.0, 0.0)

        # 4 wheels: FL, FR, RL, RR
        self.wheel_angles = [0.0, 0.0, 0.0, 0.0]  # steer angle
        self.wheel_speeds = [0.0, 0.0, 0.0, 0.0]  # rad/s

        self.is_colliding: bool = False

    @property
    def speed(self) -> float:
        """Longitudinal forward speed (m/s)."""
        return self.vel_body.x

    @property
    def speed_total(self) -> float:
        """Total speed magnitude (m/s)."""
        return self.vel_body.length()


class VehicleModel:
    """
    Modular vehicle simulation entity.
    """
    def __init__(self, config: VehicleConfig | None = None):
        self.config = config or VehicleConfig()
        self.state = VehicleState()

        # Derived inertial properties
        self._recompute_inertials()

    def _recompute_inertials(self) -> None:
        cfg = self.config
        self.a = cfg.wheelbase * (1.0 - cfg.weight_dist_front)  # COG to front axle
        self.b = cfg.wheelbase * cfg.weight_dist_front          # COG to rear axle
        # Yaw moment of inertia Iz ~ m * (a * b)
        self.Iz = cfg.mass * (self.a * self.b)
        self.wheel_radius = 0.34  # meters

    def reset(self, pos: Vec3, yaw: float, initial_speed: float = 0.0) -> None:
        """Resets vehicle to a given pose and speed."""
        self.state = VehicleState()
        self.state.pos = Vec3(pos.x, pos.y, pos.z)
        self.state.yaw = normalize_angle(yaw)
        self.state.vel_body = Vec2(initial_speed, 0.0)
        fwd = Vec2(math.cos(yaw), math.sin(yaw))
        self.state.vel_world = fwd * initial_speed

    def step(self, steering_cmd: float, throttle_cmd: float, brake_cmd: float, dt: float, surface_friction: float = 1.0) -> None:
        """
        Advances the vehicle dynamics by dt seconds.
        Control inputs:
            steering_cmd: [-1.0, 1.0] (negative = left, positive = right)
            throttle_cmd: [0.0, 1.0]
            brake_cmd:    [0.0, 1.0]
        """
        cfg = self.config
        st = self.state

        # 1. Steering rate limiting
        # Convention: steering_cmd in [-1.0, 1.0] where negative = left, positive = right.
        # In counter-clockwise Cartesian coordinates, turning left corresponds to positive steer/yaw.
        target_steer = -clamp(steering_cmd, -1.0, 1.0) * cfg.max_steering_angle
        steer_diff = target_steer - st.steering_angle
        max_delta = cfg.steering_rate * dt
        st.steering_angle += clamp(steer_diff, -max_delta, max_delta)

        # 2. Input clamp
        st.throttle = clamp(throttle_cmd, 0.0, 1.0)
        st.brake = clamp(brake_cmd, 0.0, 1.0)

        vx = st.vel_body.x
        vy = st.vel_body.y
        wz = st.yaw_rate

        # 3. Longitudinal powertrain and resistance forces
        # Drive force
        speed_factor = max(0.0, 1.0 - (vx / max(1.0, cfg.top_speed)))
        f_drive = st.throttle * cfg.max_drive_force * speed_factor

        # Braking force opposing motion
        f_brake = 0.0
        if st.brake > 0.0:
            if abs(vx) > 0.05:
                f_brake = st.brake * cfg.max_brake_force * (1.0 if vx > 0 else -1.0)
            else:
                f_brake = 0.0
                vx = 0.0

        # Aerodynamic drag: 0.5 * rho * Cd * A * vx^2
        f_drag = 0.5 * cfg.air_density * cfg.drag_coeff * cfg.frontal_area * vx * abs(vx)

        # Rolling resistance
        g = 9.81
        f_roll = cfg.rolling_resistance * cfg.mass * g * (1.0 if vx > 0.05 else (-1.0 if vx < -0.05 else 0.0))

        # Total net longitudinal force
        f_long = f_drive - f_brake - f_drag - f_roll

        # 4. Low-speed vs High-speed Lateral Dynamics
        # At very low speeds (< 1.5 m/s), dynamic tire slip models have 1/vx singularity.
        # We blend smoothly between kinematic bicycle model and dynamic tire-slip model.
        kinematic_blend = clamp((1.5 - abs(vx)) / 1.5, 0.0, 1.0) if abs(vx) < 1.5 else 0.0

        mu = cfg.tire_friction * surface_friction
        fz_front = cfg.mass * g * (self.b / cfg.wheelbase)
        fz_rear = cfg.mass * g * (self.a / cfg.wheelbase)

        if kinematic_blend < 0.99:
            # Dynamic slip model
            # Slip angles:
            alpha_f = math.atan2(vy + self.a * wz, max(0.5, abs(vx))) - st.steering_angle
            alpha_r = math.atan2(vy - self.b * wz, max(0.5, abs(vx)))

            # Nonlinear brush/hyperbolic tire lateral forces
            fy_f = -mu * fz_front * math.tanh((cfg.cornering_stiffness_front * alpha_f) / max(1.0, mu * fz_front))
            fy_r = -mu * fz_rear * math.tanh((cfg.cornering_stiffness_rear * alpha_r) / max(1.0, mu * fz_rear))

            # Dynamic accelerations
            ax_dyn = (f_long - fy_f * math.sin(st.steering_angle)) / cfg.mass + vy * wz
            ay_dyn = (fy_f * math.cos(st.steering_angle) + fy_r) / cfg.mass - vx * wz
            alpha_z_dyn = (self.a * fy_f * math.cos(st.steering_angle) - self.b * fy_r) / self.Iz
        else:
            ax_dyn = f_long / cfg.mass
            ay_dyn = 0.0
            alpha_z_dyn = 0.0

        if kinematic_blend > 0.01:
            # Kinematic bicycle model at low speeds
            # wz_kin = vx * tan(delta) / L
            wz_kin = vx * math.tan(st.steering_angle) / cfg.wheelbase
            ay_kin = 0.0
            ax_kin = f_long / cfg.mass

            # Blend
            ax = (1.0 - kinematic_blend) * ax_dyn + kinematic_blend * ax_kin
            ay = (1.0 - kinematic_blend) * ay_dyn + kinematic_blend * ay_kin
            alpha_z = (1.0 - kinematic_blend) * alpha_z_dyn + kinematic_blend * ((wz_kin - wz) / max(0.01, dt))
        else:
            ax = ax_dyn
            ay = ay_dyn
            alpha_z = alpha_z_dyn

        # 5. Numerical integration (Semi-implicit Euler)
        vx += ax * dt
        vy += ay * dt
        wz += alpha_z * dt

        # Prevent creep when stopped with brake
        if st.brake > 0.1 and abs(vx) < 0.1:
            vx = 0.0
            vy = 0.0
            wz = 0.0

        st.vel_body = Vec2(vx, vy)
        st.yaw_rate = wz
        st.accel_body = Vec2(ax, ay)

        # 6. World-frame position update
        st.yaw = normalize_angle(st.yaw + wz * dt)
        cos_yaw = math.cos(st.yaw)
        sin_yaw = math.sin(st.yaw)

        # World velocity = Rotation(yaw) * Body velocity
        vx_world = cos_yaw * vx - sin_yaw * vy
        vy_world = sin_yaw * vx + cos_yaw * vy
        st.vel_world = Vec2(vx_world, vy_world)

        st.pos.x += vx_world * dt
        st.pos.y += vy_world * dt

        # World acceleration
        st.accel_world = Vec2(cos_yaw * ax - sin_yaw * ay, sin_yaw * ax + cos_yaw * ay)

        # 7. Wheels visualization angles and rotation
        st.wheel_angles[0] = st.steering_angle  # FL
        st.wheel_angles[1] = st.steering_angle  # FR
        st.wheel_angles[2] = 0.0                # RL
        st.wheel_angles[3] = 0.0                # RR

        spin_rate = vx / self.wheel_radius
        for i in range(4):
            st.wheel_speeds[i] = (st.wheel_speeds[i] + spin_rate * dt) % (2.0 * math.pi)

    def get_obb(self) -> OBB2D:
        """Returns the oriented bounding box of the vehicle in 2D."""
        cfg = self.config
        st = self.state
        center_2d = Vec2(st.pos.x, st.pos.y)
        return OBB2D(
            center=center_2d,
            half_length=cfg.length * 0.5,
            half_width=cfg.width * 0.5,
            yaw=st.yaw
        )

    def get_wheel_transforms(self) -> List[Tuple[Vec3, float]]:
        """
        Returns the (position, steer_angle) for each of the 4 wheels:
        [FL, FR, RL, RR] in world coordinates.
        """
        cfg = self.config
        st = self.state

        cos_yaw = math.cos(st.yaw)
        sin_yaw = math.sin(st.yaw)

        hw = cfg.track_width * 0.5
        wheel_offsets = [
            (self.a, hw, st.wheel_angles[0]),    # FL
            (self.a, -hw, st.wheel_angles[1]),   # FR
            (-self.b, hw, st.wheel_angles[2]),   # RL
            (-self.b, -hw, st.wheel_angles[3]),  # RR
        ]

        results = []
        for fwd_offset, lat_offset, steer in wheel_offsets:
            wx = st.pos.x + cos_yaw * fwd_offset - sin_yaw * lat_offset
            wy = st.pos.y + sin_yaw * fwd_offset + cos_yaw * lat_offset
            wz = st.pos.z - 0.05
            results.append((Vec3(wx, wy, wz), steer))

        return results
