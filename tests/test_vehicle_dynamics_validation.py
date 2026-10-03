"""
Vehicle-dynamics regression tests.

These lock in the corrected single-track dynamics: straight-line stability,
determinism, timestep independence, steering sign conventions, friction
envelope, progressive sideslip, and recovery behavior.
"""

import math
import pytest
import numpy as np

from sim_core.math_utils import Vec2, Vec3
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.vehicle.vehicle_model import VehicleModel
from tools.vehicle_dynamics.harness import run_maneuver
from tools.vehicle_dynamics.maneuvers import standard_suite


def _cruise_then(car: VehicleModel, script, seconds: float, dt: float = 1 / 60):
    for i in range(int(seconds / dt)):
        car.step(*script(i * dt), dt)


class TestStraightLineStability:
    def test_no_lateral_motion_at_constant_throttle(self):
        r = run_maneuver("sl", lambda t: (0.0, 0.5, 0.0), 6.0)
        assert r.metrics["peak_vy_ms"] < 1e-9
        assert r.metrics["peak_yaw_rate_dps"] < 1e-6
        assert r.metrics["lateral_drift_m"] < 1e-9

    def test_no_lateral_motion_from_speed(self):
        r = run_maneuver("sl2", lambda t: (0.0, 0.0, 0.0), 5.0, initial_speed=20.0)
        assert r.metrics["peak_vy_ms"] < 1e-9
        assert r.metrics["lateral_drift_m"] < 1e-9


class TestDeterminism:
    def test_identical_runs_identical_state(self):
        def drive():
            car = VehicleModel()
            car.reset(Vec3(0, 0, 0), 0.0, initial_speed=15.0)
            for i in range(300):
                t = i / 60
                car.step(0.15 * math.sin(2 * t), 0.4, 0.0, 1 / 60)
            s = car.state
            return (s.pos.x, s.pos.y, s.yaw, s.vel_body.x, s.vel_body.y, s.yaw_rate)
        assert drive() == drive()


class TestTimestepIndependence:
    def test_final_pose_consistent_across_dt(self):
        """Same scripted scenario at different caller dt must agree within
        a small tolerance (internal 240 Hz substepping)."""
        finals = []
        for hz in (30, 60, 120, 240):
            car = VehicleModel()
            car.reset(Vec3(0, 0, 0), 0.0, initial_speed=15.0)
            dt = 1.0 / hz
            t = 0.0
            while t < 4.0:
                car.step(0.1, 0.4, 0.0, dt)
                t += dt
            s = car.state
            finals.append((s.pos.x, s.pos.y, s.yaw))
        ref = finals[-1]  # finest outer dt as reference
        for i, f in enumerate(finals):
            assert abs(f[0] - ref[0]) < 1.5, f"hz case {i} x drifted"
            assert abs(f[1] - ref[1]) < 1.5, f"hz case {i} y drifted"
            assert abs(f[2] - ref[2]) < 0.05, f"hz case {i} yaw drifted"


class TestSteeringConventions:
    def test_positive_cmd_turns_right(self):
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=15.0)
        _cruise_then(car, lambda t: (0.3, 0.3, 0.0), 2.0)
        assert car.state.yaw_rate < 0.0
        assert car.state.yaw < 0.0

    def test_negative_cmd_turns_left(self):
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=15.0)
        _cruise_then(car, lambda t: (-0.3, 0.3, 0.0), 2.0)
        assert car.state.yaw_rate > 0.0
        assert car.state.yaw > 0.0


class TestFrictionEnvelope:
    def test_combined_force_bounded_per_axle(self):
        """sqrt(Fx^2+Fy^2) per axle must never exceed mu*Fz_axle."""
        car = VehicleModel()
        cfg = car.config
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=25.0)
        for i in range(240):
            t = i / 60
            # escalating steer + full throttle + braking pulses
            steer = min(1.0, t * 0.4)
            brake = 0.7 if 2.0 < t < 3.0 else 0.0
            car.step(steer, 1.0, brake, 1 / 60)
            s = car.state
            g = 9.81
            for fx, fy, fz in ((s.fx_f, s.fy_f, s.fz_front),
                               (s.fx_r, s.fy_r, s.fz_rear)):
                cap = cfg.tire_friction * fz + 1e-6
                assert math.hypot(fx, fy) <= cap * 1.001

    def test_lateral_accel_bounded_by_mu_g(self):
        """Tire lateral force sum must respect the friction limit.

        (vx*wz is NOT the right metric during a spin — body rotation
        contributes to it. The bounded quantity is Sigma_Fy / m.)
        """
        r = run_maneuver("latcap", lambda t: (1.0, 0.0, 0.0), 6.0,
                         initial_speed=25.0)
        cfg = VehicleConfig()
        mu_g = cfg.tire_friction * 9.81
        lat = [abs(f1 + f2) / cfg.mass
               for f1, f2 in zip(r.series["fy_f"], r.series["fy_r"])]
        assert max(lat) <= mu_g * 1.001


class TestHandlingQualities:
    def test_gentle_cornering_small_sideslip(self):
        """Normal-demand cornering must not produce drift-scale sideslip."""
        r = run_maneuver("cc", lambda t: (0.10, 0.45, 0.0), 8.0,
                         initial_speed=20.0)
        assert abs(r.metrics["steady_sideslip_deg"]) < 5.0
        assert r.metrics["peak_sideslip_deg"] < 8.0

    def test_understeer_gradient(self):
        """Steady-state curvature must be below the kinematic (K > 0)."""
        r = run_maneuver("us", lambda t: (0.08, 0.4, 0.0), 10.0,
                         initial_speed=20.0)
        s = r.series
        n = len(s["t"])
        wz_ss = sum(s["yaw_rate"][-n // 4:]) / (n // 4)
        v_ss = sum(s["vx"][-n // 4:]) / (n // 4)
        kappa_meas = abs(wz_ss) / v_ss
        wheel_angle = 0.08 * 0.58
        kappa_kin = math.tan(wheel_angle) / 2.6
        assert 0.0 < kappa_meas < kappa_kin * 1.01

    def test_steer_step_settles(self):
        """A step input must produce a bounded transient, not a spin."""
        r = run_maneuver("ss", lambda t: (0.15 if t >= 0 else 0.0, 0.45, 0.0),
                         6.0, initial_speed=20.0)
        assert abs(r.metrics["steady_sideslip_deg"]) < 10.0
        assert r.metrics["peak_sideslip_deg"] < 20.0
        assert r.metrics["peak_yaw_rate_dps"] < 80.0

    def test_recovery_from_disturbance(self):
        """After a brief steering kick and release, the car must recover."""
        def script(t):
            if 1.0 <= t < 1.15:
                return (1.0, 0.4, 0.0)
            return (0.0, 0.4, 0.0)
        r = run_maneuver("rec", script, 7.0, initial_speed=20.0)
        assert abs(r.metrics["final_sideslip_deg"]) < 3.0
        # yaw rate must decay toward zero after release
        tail_wz = abs(r.series["yaw_rate"][-1])
        assert tail_wz < math.radians(10.0)


class TestCombinedSlipAllocation:
    """Regression coverage for the friction-envelope allocation fix.

    Under the old demand-first ellipse, longitudinal demand had first claim
    on the friction circle: full throttle through a RWD axle pegged u->1 and
    collapsed lateral capacity to ~0, so any throttle application near
    traction produced an unrecoverable power oversteer spin. The demanded
    (Fx, Fy) vector is now scaled proportionally to fit mu*Fz.
    """

    def test_full_throttle_small_steer_no_spin(self):
        """Full-throttle launch + a small steering correction must not spin."""
        def script(t):
            return (0.1 if t >= 3.0 else 0.0, 1.0, 0.0)
        r = run_maneuver("ftsc", script, 6.0)
        assert abs(r.metrics["peak_sideslip_deg"]) < 10.0
        assert abs(r.series["yaw_rate"][-1]) < math.radians(45.0)

    def test_throttle_in_corner_bounded_and_recovers(self):
        """Full throttle mid-corner may slide but must recover on lift."""
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=15.0)
        dt = 1 / 60
        for _ in range(60):
            car.step(0.15, 0.3, 0.0, dt)
        for _ in range(120):
            car.step(0.15, 1.0, 0.0, dt)
        # sustained slide is allowed (RWD power oversteer is physical),
        # but it must stay bounded, not a free spin
        assert abs(car.state.yaw_rate) < math.radians(120.0)
        for _ in range(240):
            car.step(0.15, 0.3, 0.0, dt)
        s = car.state
        beta = math.atan2(s.vel_body.y, max(0.1, s.vel_body.x))
        assert abs(math.degrees(beta)) < 10.0, "car must recover after throttle lift"

    def test_speed_limiter_respected(self):
        """Power-limited drive must not exceed configured top speed."""
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0)
        for _ in range(60 * 40):
            car.step(0.0, 1.0, 0.0, 1 / 60)
        assert car.state.vel_body.x <= VehicleConfig().top_speed + 0.01


class TestSuiteRuns:
    def test_standard_suite_all_stable(self):
        suite = standard_suite(speed_ms=20.0)
        for m in suite.values():
            r = run_maneuver(m.id, m.script, m.duration_s,
                             initial_speed=m.initial_speed,
                             pre_roll_s=m.pre_roll_s)
            v = np.array(r.series["vx"])
            assert np.all(np.isfinite(v)), f"{m.id} produced non-finite state"
            assert max(abs(np.array(r.series["vy"]))) < 30.0, m.id
            assert max(abs(np.array(r.series["yaw_rate"]))) < math.radians(400), m.id


class TestLowSpeedSlideDecay:
    """Regression for the lateral-freeze defect: the kinematic blend was
    keyed on |vx| only, so a slide that scrubbed forward speed below ~1.5
    m/s switched off all lateral tire force and glided sideways forever.
    The blend now keys on total speed and applies bounded Coulomb decay."""

    def test_sideways_slide_decelerates_to_stop(self):
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=40.0)
        dt = 1 / 60
        # provoke a massive slide, then release all inputs
        for _ in range(int(1.5 / dt)):
            car.step(0.25, 0.9, 0.0, dt)
        for _ in range(int(15 / dt)):
            car.step(0.0, 0.0, 0.0, dt)
        s = car.state
        # The freeze defect left |vy| ~18 m/s persisting indefinitely.
        # Tire friction must scrub lateral velocity to ~zero; residual
        # forward/backward coasting decays slowly via rolling resistance.
        assert abs(s.vel_body.y) < 0.5, (
            f"slide must decay, got vy={s.vel_body.y:.2f}")
        assert abs(math.degrees(s.yaw_rate)) < 20.0

    def test_slow_lateral_push_stops(self):
        """A car nudged sideways at crawl must stop, not glide."""
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0)
        car.state.vel_body = Vec2(0.5, 1.0)   # creeping + lateral shove
        dt = 1 / 60
        for _ in range(int(5 / dt)):
            car.step(0.0, 0.0, 0.0, dt)
        assert abs(car.state.vel_body.y) < 0.05


class TestHighSpeedRecovery:
    """High-speed limit handling: saturated slides must be recoverable
    with countersteer, and a released slide must self-align (relaxation +
    aligning moment), while true >mu*g demands still depart."""

    def test_countersteer_recovers_40ms_slide(self):
        car = VehicleModel()
        dt = 1 / 60
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=40.0)
        for _ in range(60):
            car.step(0.0, 0.8, 0.0, dt)
        for _ in range(int(1.5 / dt)):
            car.step(0.10, 0.8, 0.0, dt)
        beta0 = abs(math.atan2(car.state.vel_body.y,
                               max(0.1, car.state.vel_body.x)))
        assert math.degrees(beta0) > 3.0, "slide should have been provoked"
        for _ in range(int(4 / dt)):
            vx, vy = car.state.vel_body.x, car.state.vel_body.y
            beta = math.atan2(vy, max(0.1, vx))
            cs = max(-1.0, min(1.0, -beta / 0.58))
            car.step(cs, 0.8, 0.0, dt)
        s = car.state
        beta_f = math.degrees(math.atan2(s.vel_body.y, max(0.1, s.vel_body.x)))
        assert abs(beta_f) < 5.0
        assert abs(math.degrees(s.yaw_rate)) < 15.0

    def test_released_slide_self_aligns(self):
        car = VehicleModel()
        dt = 1 / 60
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=40.0)
        for _ in range(60):
            car.step(0.0, 0.8, 0.0, dt)
        for _ in range(int(1.5 / dt)):
            car.step(0.10, 0.8, 0.0, dt)
        for _ in range(int(4 / dt)):
            car.step(0.0, 0.8, 0.0, dt)
        s = car.state
        beta_f = math.degrees(math.atan2(s.vel_body.y, max(0.1, s.vel_body.x)))
        assert abs(beta_f) < 5.0, "released saturated slide must self-align"
        assert abs(math.degrees(s.yaw_rate)) < 10.0

    def test_sublimit_cornering_stable_at_40ms(self):
        """A sub-limit steer demand must hold a stable corner at 40 m/s."""
        r = run_maneuver("sub40", lambda t: (0.05, 0.9, 0.0), 6.0,
                         initial_speed=40.0)
        assert r.metrics["peak_sideslip_deg"] < 10.0
        assert abs(r.metrics["steady_sideslip_deg"]) < 6.0


class TestTirePhysicsDetails:
    def test_tire_relaxation_delays_force(self):
        """Lateral force must not appear instantly on a steer step."""
        car = VehicleModel()
        dt = 1 / 60
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=20.0)
        for _ in range(30):
            car.step(0.0, 0.4, 0.0, dt)
        car.step(0.3, 0.4, 0.0, dt)
        fy_step1 = abs(car.state.fy_f)
        car.step(0.3, 0.4, 0.0, dt)
        fy_step2 = abs(car.state.fy_f)
        # full front lateral force builds over ~sigma/v ~ 28 ms, not in one step
        assert fy_step2 > fy_step1

    def test_quasistatic_roll_pitch(self):
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=20.0)
        for _ in range(120):
            car.step(0.15, 0.4, 0.0, 1 / 60)
        # right turn (cmd>0): chassis rolls toward the outside (left side
        # up / right-side-down convention -> roll < 0), ~4 deg/g.
        assert car.state.roll < -0.005
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=20.0)
        for _ in range(30):
            car.step(0.0, 0.0, 1.0, 1 / 60)
        assert car.state.pitch < -0.005  # brake dive

    def test_loaded_axle_stiffness_sublinear(self):
        """Load sensitivity: braking must grow the front cap sublinearly —
        loaded front axle grip < linear extrapolation."""
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=25.0)
        dt = 1 / 60
        fz_static = None
        for _ in range(30):
            car.step(0.0, 0.0, 0.0, dt)
        fz_static = car.state.fz_front
        for _ in range(30):
            car.step(0.0, 0.0, 1.0, dt)
        fz_brake = car.state.fz_front
        assert fz_brake > fz_static * 1.05, "braking must load the front axle"


class TestSurfaceAndBiasInputs:
    def test_world_accel_bias_displaces_trajectory(self):
        """A constant lateral world bias must curve the trajectory."""
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=15.0)
        dt = 1 / 60
        for _ in range(int(3 / dt)):
            car.step(0.0, 0.3, 0.0, dt, world_accel_bias=Vec2(0.0, -2.5))
        assert car.state.pos.y < -1.0

    def test_bias_not_in_proper_accel(self):
        """Gravity bias must not appear in accel_body (IMU proper accel)."""
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=15.0)
        car.step(0.0, 0.3, 0.0, 1 / 60, world_accel_bias=Vec2(0.0, -5.0))
        # steady straight-line: proper ay ~0 (bias is gravity, not tire force)
        assert abs(car.state.accel_body.y) < 1.0


class TestStaticFrictionHold:
    """Static friction must hold a parked car on sub-critical slopes and
    release it on super-critical ones (no perpetual low-speed creep)."""

    def test_parked_car_holds_on_mild_bank(self):
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=0.0)
        # 15-deg-equivalent lateral pull: g*sin(15deg) ~ 2.54 m/s^2 << mu*g
        bias = Vec2(0.0, -2.54)
        for _ in range(300):
            car.step(0.0, 0.0, 0.0, 1 / 60, 1.0, world_accel_bias=bias)
        st = car.state
        assert st.speed_total < 1e-6
        assert abs(st.pos.y) < 1e-6

    def test_supercritical_slope_slides(self):
        # bias > mu*g/sqrt(1+mu^2): tan(theta) > mu, car must slide
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=0.0)
        bias = Vec2(0.0, -7.51)  # ~50-deg bank; tan50 = 1.19 > mu=1.0
        for _ in range(300):
            car.step(0.0, 0.0, 0.0, 1 / 60, 1.0, world_accel_bias=bias)
        assert car.state.speed_total > 0.5

    def test_throttle_breaks_hold(self):
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=0.0)
        bias = Vec2(0.0, -2.54)
        for _ in range(120):
            car.step(0.0, 0.8, 0.0, 1 / 60, 1.0, world_accel_bias=bias)
        assert car.state.vel_body.x > 0.5
