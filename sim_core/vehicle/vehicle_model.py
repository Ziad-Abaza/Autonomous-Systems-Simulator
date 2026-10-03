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
        'is_colliding',
        'alpha_f', 'alpha_r', 'fy_f', 'fy_r',
        'fx_f', 'fx_r', 'fz_front', 'fz_rear'
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

        # Per-step tire diagnostics (validation/telemetry only — not part of
        # any RL observation schema)
        self.alpha_f: float = 0.0     # front axle slip angle (rad)
        self.alpha_r: float = 0.0     # rear axle slip angle (rad)
        self.fy_f: float = 0.0        # front axle lateral force (N)
        self.fy_r: float = 0.0        # rear axle lateral force (N)
        self.fx_f: float = 0.0        # front axle longitudinal tire force (N)
        self.fx_r: float = 0.0        # rear axle longitudinal tire force (N)
        self.fz_front: float = 0.0    # front axle normal load (N)
        self.fz_rear: float = 0.0     # rear axle normal load (N)

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
        # Yaw inertia: explicit override wins, else uniform-mass estimate m*a*b
        self.Iz = cfg.yaw_inertia if cfg.yaw_inertia > 0.0 else cfg.mass * (self.a * self.b)
        self.wheel_radius = cfg.wheel_radius

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

        Internally integrates at `physics_substeps` per call so results are
        stable and nearly independent of the caller's dt.
        """
        cfg = self.config
        st = self.state

        # Control inputs are held constant across the substeps of one call.
        # Convention: steering_cmd negative = left. In CCW-positive Cartesian
        # coordinates a left turn is positive wheel angle / yaw rate.
        target_steer = -clamp(steering_cmd, -1.0, 1.0) * cfg.max_steering_angle
        st.throttle = clamp(throttle_cmd, 0.0, 1.0)
        st.brake = clamp(brake_cmd, 0.0, 1.0)

        n_sub = max(1, int(cfg.physics_substeps))
        h = dt / n_sub
        for _ in range(n_sub):
            self._substep(target_steer, h, surface_friction)

        # Wheel visualization angles and rotation (non-physical bookkeeping)
        st.wheel_angles[0] = st.steering_angle  # FL
        st.wheel_angles[1] = st.steering_angle  # FR
        st.wheel_angles[2] = 0.0                # RL
        st.wheel_angles[3] = 0.0                # RR
        spin_rate = st.vel_body.x / self.wheel_radius
        for i in range(4):
            st.wheel_speeds[i] = (st.wheel_speeds[i] + spin_rate * dt) % (2.0 * math.pi)

    def _substep(self, target_steer: float, h: float, surface_friction: float) -> None:
        """Integrates the single-track model by one substep h."""
        cfg = self.config
        st = self.state
        g = 9.81

        # 1. Steering rate limiting (road-wheel angle, rad)
        steer_diff = target_steer - st.steering_angle
        max_delta = cfg.steering_rate * h
        st.steering_angle += clamp(steer_diff, -max_delta, max_delta)

        vx = st.vel_body.x
        vy = st.vel_body.y
        wz = st.yaw_rate

        # 2. Longitudinal tire forces (per axle) + free-body losses.
        #    Drive/brake forces act at the tire contact patches and consume
        #    friction budget; drag/rolling resistance act on the body.
        speed_factor = max(0.0, 1.0 - (vx / max(1.0, cfg.top_speed)))
        f_drive = st.throttle * cfg.max_drive_force * speed_factor

        f_brake = 0.0
        if st.brake > 0.0:
            if abs(vx) > 0.05:
                f_brake = st.brake * cfg.max_brake_force * (1.0 if vx > 0 else -1.0)
            else:
                vx = 0.0

        f_drag = 0.5 * cfg.air_density * cfg.drag_coeff * cfg.frontal_area * vx * abs(vx)
        f_roll = cfg.rolling_resistance * cfg.mass * g * (1.0 if vx > 0.05 else (-1.0 if vx < -0.05 else 0.0))

        # Per-axle longitudinal tire force (positive = forward on vehicle)
        fx_f = f_drive * cfg.drive_force_front_fraction - f_brake * cfg.brake_bias_front
        fx_r = f_drive * (1.0 - cfg.drive_force_front_fraction) - f_brake * (1.0 - cfg.brake_bias_front)

        # 3. Normal loads: static + longitudinal load transfer from CG height.
        #    accel_body stores proper acceleration (specific force, Sigma_F/m)
        #    which is exactly the quantity that transfers load — no Coriolis
        #    correction needed.
        #    Lateral transfer is omitted: this single-track model aggregates
        #    each axle to one tire force, so it would not change axle sums.
        mu = cfg.tire_friction * surface_friction
        fz_f_static = cfg.mass * g * (self.b / cfg.wheelbase)
        fz_r_static = cfg.mass * g * (self.a / cfg.wheelbase)
        transfer = cfg.mass * st.accel_body.x * cfg.cog_height / cfg.wheelbase
        transfer = clamp(transfer, -0.9 * fz_r_static, 0.9 * fz_f_static)
        fz_front = fz_f_static - transfer   # braking (ax<0) loads the front
        fz_rear = fz_r_static + transfer

        # 4. Tire model: saturating-tanh lateral force with per-axle
        #    friction-ellipse coupling. Longitudinal demand consumes grip
        #    first; remaining grip sqrt(1-u^2)*muFz bounds lateral force.
        vx_eff = vx if abs(vx) >= 0.5 else (0.5 if vx >= 0.0 else -0.5)
        alpha_f = math.atan2(vy + self.a * wz, vx_eff) - st.steering_angle
        alpha_r = math.atan2(vy - self.b * wz, vx_eff)

        cap_f = mu * fz_front
        cap_r = mu * fz_rear
        # Clamp longitudinal tire force to available friction
        u_f = clamp(fx_f / max(1.0, cap_f), -1.0, 1.0)
        u_r = clamp(fx_r / max(1.0, cap_r), -1.0, 1.0)
        fx_f = u_f * cap_f
        fx_r = u_r * cap_r
        lat_cap_f = cap_f * math.sqrt(max(0.0, 1.0 - u_f * u_f))
        lat_cap_r = cap_r * math.sqrt(max(0.0, 1.0 - u_r * u_r))

        low_thr = cfg.low_speed_threshold
        kinematic_blend = clamp((low_thr - abs(vx)) / low_thr, 0.0, 1.0) if abs(vx) < low_thr else 0.0

        if kinematic_blend < 0.99:
            fy_f = -lat_cap_f * math.tanh(cfg.cornering_stiffness_front * alpha_f / max(1.0, cap_f))
            fy_r = -lat_cap_r * math.tanh(cfg.cornering_stiffness_rear * alpha_r / max(1.0, cap_r))

            f_x_total = fx_f + fx_r - f_drag - f_roll
            ax_dyn = (f_x_total - fy_f * math.sin(st.steering_angle)) / cfg.mass + vy * wz
            ay_dyn = (fy_f * math.cos(st.steering_angle) + fy_r) / cfg.mass - vx * wz
            alpha_z_dyn = (self.a * fy_f * math.cos(st.steering_angle) - self.b * fy_r) / self.Iz
        else:
            fy_f = fy_r = 0.0
            ax_dyn = (fx_f + fx_r - f_drag - f_roll) / cfg.mass
            ay_dyn = 0.0
            alpha_z_dyn = 0.0

        if kinematic_blend > 0.01:
            # Low-speed regularization: blend toward kinematic bicycle model.
            # The yaw-rate term is a bounded relaxation (time constant tau),
            # not deadbeat forcing.
            wz_kin = vx * math.tan(st.steering_angle) / cfg.wheelbase
            f_x_total = fx_f + fx_r - f_drag - f_roll
            ax_kin = f_x_total / cfg.mass
            ax = (1.0 - kinematic_blend) * ax_dyn + kinematic_blend * ax_kin
            ay = (1.0 - kinematic_blend) * ay_dyn
            relax = (wz_kin - wz) / max(cfg.low_speed_relax_tau, h)
            alpha_z = (1.0 - kinematic_blend) * alpha_z_dyn + kinematic_blend * relax
        else:
            ax = ax_dyn
            ay = ay_dyn
            alpha_z = alpha_z_dyn

        # 5. Integration (explicit Euler at substep rate)
        vx += ax * h
        vy += ay * h
        wz += alpha_z * h

        # Prevent creep when stopped with brake
        if st.brake > 0.1 and abs(vx) < 0.1:
            vx = 0.0
            vy = 0.0
            wz = 0.0

        st.vel_body = Vec2(vx, vy)
        st.yaw_rate = wz
        # accel_body is PROPER acceleration (specific force Sigma_F/m), i.e.
        # what an accelerometer measures — d(vx,vy)/dt with the frame-rotation
        # terms removed. In steady cornering it correctly reads centripetal
        # acceleration instead of decaying to zero.
        proper_ax = ax - vy * wz
        proper_ay = ay + vx * wz
        st.accel_body = Vec2(proper_ax, proper_ay)

        # 6. World-frame pose update
        st.yaw = normalize_angle(st.yaw + wz * h)
        cos_yaw = math.cos(st.yaw)
        sin_yaw = math.sin(st.yaw)
        vx_world = cos_yaw * vx - sin_yaw * vy
        vy_world = sin_yaw * vx + cos_yaw * vy
        st.vel_world = Vec2(vx_world, vy_world)
        st.pos.x += vx_world * h
        st.pos.y += vy_world * h
        st.accel_world = Vec2(cos_yaw * proper_ax - sin_yaw * proper_ay,
                              sin_yaw * proper_ax + cos_yaw * proper_ay)

        # 7. Diagnostics (last substep state — telemetry/validation only)
        st.alpha_f = alpha_f
        st.alpha_r = alpha_r
        st.fy_f = fy_f
        st.fy_r = fy_r
        st.fx_f = fx_f
        st.fx_r = fx_r
        st.fz_front = fz_front
        st.fz_rear = fz_rear

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
