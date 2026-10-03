"""
Vehicle-dynamics regression tests.

These lock in the corrected single-track dynamics: straight-line stability,
determinism, timestep independence, steering sign conventions, friction
envelope, progressive sideslip, and recovery behavior.
"""

import math
import pytest
import numpy as np

from sim_core.math_utils import Vec3
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
