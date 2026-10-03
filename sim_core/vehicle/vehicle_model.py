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
        'fx_f', 'fx_r', 'fz_front', 'fz_rear',
        'alpha_f_eff', 'alpha_r_eff',
        'alpha_w_eff', 'fy_wheels', 'fx_wheels', 'fz_wheels'
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

        # Tire relaxation lag state: effective slip angles that the tire
        # force actually responds to (first-order lag on measured slip).
        self.alpha_f_eff: float = 0.0
        self.alpha_r_eff: float = 0.0
        # Per-wheel diagnostics (pacejka4 mode): FL, FR, RL, RR
        self.alpha_w_eff = [0.0, 0.0, 0.0, 0.0]
        self.fy_wheels = [0.0, 0.0, 0.0, 0.0]
        self.fx_wheels = [0.0, 0.0, 0.0, 0.0]
        self.fz_wheels = [0.0, 0.0, 0.0, 0.0]

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

    def step(self, steering_cmd: float, throttle_cmd: float, brake_cmd: float, dt: float,
             surface_friction: float = 1.0,
             world_accel_bias: Vec2 | None = None) -> None:
        """
        Advances the vehicle dynamics by dt seconds.
        Control inputs:
            steering_cmd: [-1.0, 1.0] (negative = left, positive = right)
            throttle_cmd: [0.0, 1.0]
            brake_cmd:    [0.0, 1.0]
            surface_friction: local surface mu multiplier (track/region dependent)
            world_accel_bias: optional world-frame acceleration (m/s^2) applied
                to the chassis — used to inject gravity components from road
                grade and banking. It is a body force: it integrates velocity
                but is NOT part of `accel_body` (proper acceleration / IMU).

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
            self._substep(target_steer, h, surface_friction, world_accel_bias)

        # Quasi-static suspension attitude from proper acceleration.
        # roll > 0 = right side down (rolls outside in a left turn);
        # pitch > 0 = nose up (squat under acceleration, dive under braking).
        st.roll = math.radians(cfg.roll_deg_per_g) * (st.accel_body.y / 9.81)
        st.pitch = math.radians(cfg.pitch_deg_per_g) * (st.accel_body.x / 9.81)

        # Wheel visualization angles and rotation (non-physical bookkeeping)
        st.wheel_angles[0] = st.steering_angle  # FL
        st.wheel_angles[1] = st.steering_angle  # FR
        st.wheel_angles[2] = 0.0                # RL
        st.wheel_angles[3] = 0.0                # RR
        spin_rate = st.vel_body.x / self.wheel_radius
        for i in range(4):
            st.wheel_speeds[i] = (st.wheel_speeds[i] + spin_rate * dt) % (2.0 * math.pi)

    def _pacejka4(self, vx: float, vy: float, wz: float, vx_eff: float,
                  fz_front: float, fz_rear: float,
                  fz_f_static: float, fz_r_static: float, mu: float,
                  ca_f_axle: float, ca_r_axle: float,
                  fx_f_axle: float, fx_r_axle: float,
                  h: float, v_abs: float) -> Tuple[float, float]:
        """
        4-wheel magic-formula tire model.

        Per corner: normal load (static + longitudinal + lateral transfer),
        wheel-local slip angle (includes yaw-rate velocity offsets), and a
        Pacejka lateral curve Fy = -D*sin(C*atan(B*alpha)) — unlike the
        saturating tanh this genuinely decays past peak slip. Longitudinal
        force shares the per-wheel friction envelope. Returns
        (trail_f, trail_r) pneumatic-trail factors for the yaw equation and
        writes per-wheel + axle diagnostics into the state.
        """
        cfg = self.config
        st = self.state
        hw = cfg.track_width * 0.5

        # Lateral load transfer splits by roll-stiffness distribution.
        latT = cfg.mass * st.accel_body.y * cfg.cog_height / cfg.track_width
        latT_f = latT * cfg.roll_stiffness_front_fraction
        latT_r = latT * (1.0 - cfg.roll_stiffness_front_fraction)

        # Order: FL, FR, RL, RR. y < 0 is the left side.
        wheels = (
            (self.a, -hw, fz_front * 0.5 + latT_f, st.steering_angle,
             fz_f_static * 0.5, ca_f_axle * 0.5),
            (self.a, hw, fz_front * 0.5 - latT_f, st.steering_angle,
             fz_f_static * 0.5, ca_f_axle * 0.5),
            (-self.b, -hw, fz_rear * 0.5 + latT_r, 0.0,
             fz_r_static * 0.5, ca_r_axle * 0.5),
            (-self.b, hw, fz_rear * 0.5 - latT_r, 0.0,
             fz_r_static * 0.5, ca_r_axle * 0.5),
        )
        fx_each = (fx_f_axle * 0.5, fx_f_axle * 0.5,
                   fx_r_axle * 0.5, fx_r_axle * 0.5)
        relax_k = min(1.0, max(v_abs, 0.5) * h / max(1e-3, cfg.tire_relaxation_m))
        shape_c = cfg.pacejka_shape_c

        for i, (xi, yi, fz_w, sw, fz_ref, ca_w) in enumerate(wheels):
            fz_w = max(0.0, fz_w)
            # Wheel-ground velocities: yaw rate shifts longitudinal speed by
            # -wz*y and lateral speed by +wz*x.
            vx_w = vx - wz * yi
            vy_w = vy + wz * xi
            vx_w_eff = vx_w if abs(vx_w) >= 0.5 else (0.5 if vx_w >= 0.0 else -0.5)
            alpha_w = math.atan2(vy_w, vx_w_eff) - sw
            st.alpha_w_eff[i] += relax_k * (alpha_w - st.alpha_w_eff[i])
            aw = st.alpha_w_eff[i]

            cap_w = (mu * fz_w / (1.0 + cfg.tire_load_sensitivity
                                  * max(0.0, fz_w / max(1.0, fz_ref) - 1.0))
                     if fz_w > 0.0 else 0.0)
            ca_w = ca_w * max(0.0, fz_w / max(1.0, fz_ref)) ** cfg.tire_stiffness_load_exp

            if cap_w > 1.0 and ca_w > 1.0:
                # B chosen so dFy/dα|0 = C*B*D = ca_w — small-slip slope
                # matches the axle-aggregate model's cornering stiffness.
                B = ca_w / (shape_c * cap_w)
                fy_w = -cap_w * math.sin(shape_c * math.atan(B * aw))
            else:
                fy_w = 0.0

            # Per-wheel friction-envelope coupling (combined slip).
            fx_w = fx_each[i]
            mag = math.hypot(fx_w, fy_w)
            if mag > cap_w > 0.0:
                s = cap_w / mag
                fx_w *= s
                fy_w *= s

            st.fy_wheels[i] = fy_w
            st.fx_wheels[i] = fx_w
            st.fz_wheels[i] = fz_w

        st.fy_f = st.fy_wheels[0] + st.fy_wheels[1]
        st.fy_r = st.fy_wheels[2] + st.fy_wheels[3]
        st.fx_f = st.fx_wheels[0] + st.fx_wheels[1]
        st.fx_r = st.fx_wheels[2] + st.fx_wheels[3]

        # Pneumatic trail keyed on the axle-effective slip (same semantics
        # as the aggregate model).
        trail_f = cfg.pneumatic_trail_m * max(0.0, 1.0 - abs(st.alpha_f_eff) / 0.3)
        trail_r = cfg.pneumatic_trail_m * max(0.0, 1.0 - abs(st.alpha_r_eff) / 0.3)
        return trail_f, trail_r

    def _substep(self, target_steer: float, h: float, surface_friction: float,
                 bias_w: Vec2 | None) -> None:
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
        v_abs = math.hypot(vx, vy)

        # 2. Longitudinal tire forces (per axle) + free-body losses.
        #    Drive/brake forces act at the tire contact patches and consume
        #    friction budget; drag/rolling resistance act on the body.
        #    The engine is power-limited: F = P/v at speed, capped at low
        #    speed by max_drive_force (tractive limit). This replaces the
        #    old linear (1 - v/top_speed) taper, which under-represented
        #    mid-range acceleration and amplified force in reverse.
        f_power = cfg.engine_power / max(cfg.power_min_speed, abs(vx))
        f_drive = st.throttle * min(cfg.max_drive_force, f_power)
        if vx >= 0.0:
            # Smooth speed-limiter band over the last 2 m/s to top_speed.
            f_drive *= clamp((cfg.top_speed - vx) / 2.0, 0.0, 1.0)

        f_brake = 0.0
        if st.brake > 0.0:
            if abs(vx) > 0.05:
                f_brake = st.brake * cfg.max_brake_force * (1.0 if vx > 0 else -1.0)
            else:
                vx = 0.0

        # Aerodynamic resistance opposes the velocity vector with per-axis
        # coefficients: frontal Cd*A longitudinally, broadside Cs*As
        # laterally. A car in a deep slide really does shed lateral speed to
        # aero side force — previously drag acted on vx only and a sideways
        # slide coasted aero-free.
        f_drag_x = 0.5 * cfg.air_density * cfg.drag_coeff * cfg.frontal_area * v_abs * vx
        f_drag_y = 0.5 * cfg.air_density * cfg.side_drag_coeff * cfg.side_area * v_abs * vy
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

        # Gravity bias from road grade/banking (world frame -> body frame).
        # Computed before tire loads: on a tilted road plane the normal
        # force is m*g*cos(theta) with sin(theta) = |bias|/g, so every
        # friction capacity shrinks by cos(theta). Flat road -> cos=1.
        ax_b = ay_b = 0.0
        if bias_w is not None:
            cy_b = math.cos(st.yaw)
            sy_b = math.sin(st.yaw)
            ax_b = cy_b * bias_w.x + sy_b * bias_w.y
            ay_b = -sy_b * bias_w.x + cy_b * bias_w.y
        b_mag = math.hypot(ax_b, ay_b)
        cos_tilt = math.sqrt(max(0.0, 1.0 - (b_mag / g) ** 2))

        fz_f_static = cfg.mass * g * cos_tilt * (self.b / cfg.wheelbase)
        fz_r_static = cfg.mass * g * cos_tilt * (self.a / cfg.wheelbase)
        transfer = cfg.mass * st.accel_body.x * cfg.cog_height / cfg.wheelbase
        transfer = clamp(transfer, -0.9 * fz_r_static, 0.9 * fz_f_static)
        fz_front = fz_f_static - transfer   # braking (ax<0) loads the front
        fz_rear = fz_r_static + transfer

        # 4. Tire model: saturating-tanh lateral force with per-axle
        #    friction-envelope coupling, load sensitivity, relaxation lag
        #    and pneumatic trail.
        vx_eff = vx if abs(vx) >= 0.5 else (0.5 if vx >= 0.0 else -0.5)
        alpha_f = math.atan2(vy + self.a * wz, vx_eff) - st.steering_angle
        alpha_r = math.atan2(vy - self.b * wz, vx_eff)

        # Tire relaxation length: the contact patch lags the slip input by
        # ~sigma metres of travel — a first-order lag on the effective slip
        # angle. Removes the unphysical instant force step on steer input
        # and gives the transient response a finite rise time.
        relax_k = min(1.0, max(v_abs, 0.5) * h / max(1e-3, cfg.tire_relaxation_m))
        st.alpha_f_eff += relax_k * (alpha_f - st.alpha_f_eff)
        st.alpha_r_eff += relax_k * (alpha_r - st.alpha_r_eff)

        # Load sensitivity: a real tire's capacity grows SUBlinearly with
        # load (friction coefficient degrades under overload), and its
        # cornering stiffness scales ~Fz^0.8, not linearly.
        cap_f = mu * fz_front / (1.0 + cfg.tire_load_sensitivity
                                 * max(0.0, fz_front / fz_f_static - 1.0))
        cap_r = mu * fz_rear / (1.0 + cfg.tire_load_sensitivity
                                * max(0.0, fz_rear / fz_r_static - 1.0))
        ca_f = cfg.cornering_stiffness_front * max(0.0, fz_front / fz_f_static) ** cfg.tire_stiffness_load_exp
        ca_r = cfg.cornering_stiffness_rear * max(0.0, fz_rear / fz_r_static) ** cfg.tire_stiffness_load_exp

        # The kinematic blend must key on TOTAL speed: a sideways slide that
        # scrubs vx below the threshold while vy is still large is NOT a
        # kinematic vehicle — switching the tires off there left the car
        # gliding laterally with zero friction (the "grip vanishes" bug).
        low_thr = cfg.low_speed_threshold
        kinematic_blend = clamp((low_thr - v_abs) / low_thr, 0.0, 1.0) if v_abs < low_thr else 0.0

        if kinematic_blend < 0.99:
            af_eff = st.alpha_f_eff
            ar_eff = st.alpha_r_eff
            if cfg.tire_model == "pacejka4":
                trail_f, trail_r = self._pacejka4(
                    vx, vy, wz, vx_eff, fz_front, fz_rear,
                    fz_f_static, fz_r_static, mu, ca_f, ca_r,
                    fx_f, fx_r, h, v_abs)
                # fx_f/fx_r may have been clipped by per-wheel envelopes;
                # _pacejka4 writes st.fx_f/st.fx_r — refresh locals.
                fx_f, fx_r = st.fx_f, st.fx_r
                fy_f, fy_r = st.fy_f, st.fy_r
            else:
                fy_f = -cap_f * math.tanh(ca_f * af_eff / max(1.0, cap_f))
                fy_r = -cap_r * math.tanh(ca_r * ar_eff / max(1.0, cap_r))
                trail_f = cfg.pneumatic_trail_m * max(0.0, 1.0 - abs(af_eff) / 0.3)
                trail_r = cfg.pneumatic_trail_m * max(0.0, 1.0 - abs(ar_eff) / 0.3)
            # Tire friction envelope: |F_axle| <= mu_eff*Fz_axle.
            # The contact patch produces a bounded force VECTOR whose
            # direction follows the combined-slip state — excess demand
            # becomes tire slide, so the demanded (Fx, Fy) vector is scaled
            # proportionally to fit the friction circle.
            # (pacejka4 already applied its envelopes per wheel; re-check
            # the axle sums so aggregates still respect their caps.)
            mag_f = math.hypot(fx_f, fy_f)
            if mag_f > cap_f:
                s_f = cap_f / mag_f
                fx_f *= s_f
                fy_f *= s_f
            mag_r = math.hypot(fx_r, fy_r)
            if mag_r > cap_r:
                s_r = cap_r / mag_r
                fx_r *= s_r
                fy_r *= s_r

            # Pneumatic trail: the lateral force acts behind the wheel
            # centre, producing a self-aligning moment Mz = -t*Fy that adds
            # understeer at small slip and vanishes once the patch slides.
            trail_f = cfg.pneumatic_trail_m * max(0.0, 1.0 - abs(af_eff) / 0.3)
            trail_r = cfg.pneumatic_trail_m * max(0.0, 1.0 - abs(ar_eff) / 0.3)

            f_x_total = fx_f + fx_r - f_drag_x - f_roll
            ax_dyn = (f_x_total - fy_f * math.sin(st.steering_angle)) / cfg.mass + vy * wz
            ay_dyn = (fy_f * math.cos(st.steering_angle) + fy_r - f_drag_y) / cfg.mass - vx * wz
            alpha_z_dyn = (self.a * fy_f * math.cos(st.steering_angle) - self.b * fy_r
                           - trail_f * fy_f * math.cos(st.steering_angle)
                           - trail_r * fy_r) / self.Iz
            if cfg.tire_model == "pacejka4" and st.fx_wheels:
                # Differential longitudinal tire forces left/right add a
                # real yaw moment: Mz += -sum(y_i * Fx_i).
                hw = cfg.track_width * 0.5
                fx_L = st.fx_wheels[0] + st.fx_wheels[2]
                fx_R = st.fx_wheels[1] + st.fx_wheels[3]
                alpha_z_dyn += hw * (fx_L - fx_R) / self.Iz
        else:
            fy_f = fy_r = 0.0
            fx_f = clamp(fx_f, -cap_f, cap_f)
            fx_r = clamp(fx_r, -cap_r, cap_r)
            ax_dyn = (fx_f + fx_r - f_drag_x - f_roll) / cfg.mass
            ay_dyn = 0.0
            alpha_z_dyn = 0.0

        if kinematic_blend > 0.01:
            # Low-speed regularization: blend toward kinematic bicycle model.
            # The yaw-rate term is a bounded relaxation (time constant tau),
            # not deadbeat forcing.
            # Residual lateral slide still sees Coulomb-like tire friction:
            # bounded by slide_decay*mu*g*cos(theta) and by |vy|/h so it can
            # bring vy to zero within a step but can never reverse its sign.
            # On a slope steeper than the friction angle the bias exceeds
            # this capacity and the car genuinely slides downhill.
            wz_kin = vx * math.tan(st.steering_angle) / cfg.wheelbase
            f_x_total = fx_f + fx_r - f_drag_x - f_roll
            ax_kin = f_x_total / cfg.mass
            ay_kin = (-math.copysign(min(cfg.low_speed_slide_decay * mu * g * cos_tilt,
                                        abs(vy) / h), vy)
                      if abs(vy) > 1e-9 else 0.0)
            ax = (1.0 - kinematic_blend) * ax_dyn + kinematic_blend * ax_kin
            ay = (1.0 - kinematic_blend) * ay_dyn + kinematic_blend * ay_kin
            relax = (wz_kin - wz) / max(cfg.low_speed_relax_tau, h)
            alpha_z = (1.0 - kinematic_blend) * alpha_z_dyn + kinematic_blend * relax
        else:
            ax = ax_dyn
            ay = ay_dyn
            alpha_z = alpha_z_dyn

        # Static friction hold: a parked car on a slope does not creep while
        # the gravity pull is within the tires' static capacity — the patch
        # supplies an equal-and-opposite reaction. On a plane tilted by the
        # bias, hold iff g*sin(theta) <= mu*g*cos(theta), i.e. the closed
        # form |bias| <= mu*g/sqrt(1+mu^2). Only engages at true rest with
        # no drive demand; any commanded tire force (or a steeper slope)
        # breaks the hold.
        if (v_abs < 0.3 and kinematic_blend > 0.5
                and b_mag <= mu * g / math.sqrt(1.0 + mu * mu)
                and abs(f_x_total) < 1.0):
            ax_b = ay_b = 0.0
            vx = vy = 0.0

        # 5. Integration (explicit Euler at substep rate)
        vx += (ax + ax_b) * h
        vy += (ay + ay_b) * h
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
        # terms removed and WITHOUT the gravity bias (an accelerometer cannot
        # feel gravity as specific force). In steady cornering it correctly
        # reads centripetal acceleration instead of decaying to zero.
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
            wz = st.pos.z + cfg.wheel_radius
            results.append((Vec3(wx, wy, wz), steer))

        return results
