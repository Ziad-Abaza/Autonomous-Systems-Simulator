"""
Unit tests for modular vehicle dynamics, steering, and braking.
"""

import math
import pytest
from sim_core.math_utils import Vec3
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.vehicle.vehicle_model import VehicleModel


def test_vehicle_acceleration_and_top_speed():
    cfg = VehicleConfig(top_speed=40.0)
    car = VehicleModel(cfg)
    car.reset(pos=Vec3(0, 0, 0), yaw=0.0)

    dt = 1.0 / 60.0
    # Apply full throttle for 3 seconds
    for _ in range(180):
        car.step(steering_cmd=0.0, throttle_cmd=1.0, brake_cmd=0.0, dt=dt)

    assert car.state.speed > 10.0, "Car should accelerate under 100% throttle"
    assert car.state.pos.x > 15.0, "Car should travel forward along +X"
    assert abs(car.state.pos.y) < 0.1, "Car should remain on straight line when steering=0"


def test_vehicle_braking():
    car = VehicleModel()
    car.reset(pos=Vec3(0, 0, 0), yaw=0.0, initial_speed=20.0)
    assert car.state.speed == 20.0

    dt = 1.0 / 60.0
    # Apply 100% brake for 3 seconds (180 steps)
    for _ in range(180):
        car.step(steering_cmd=0.0, throttle_cmd=0.0, brake_cmd=1.0, dt=dt)

    assert car.state.speed < 1.0, "Car should decelerate significantly under full brake"


def test_vehicle_turning_and_yaw():
    car = VehicleModel()
    car.reset(pos=Vec3(0, 0, 0), yaw=0.0, initial_speed=10.0)

    dt = 1.0 / 60.0
    # Steer left (negative cmd -> positive yaw / +Y deviation)
    for _ in range(60):
        car.step(steering_cmd=-0.6, throttle_cmd=0.5, brake_cmd=0.0, dt=dt)

    assert car.state.yaw > 0.05, "Steer left should produce positive yaw angle"
    assert car.state.pos.y > 0.5, "Steer left should produce positive lateral displacement (+Y)"

    # Reset and steer right (positive cmd -> negative yaw / -Y deviation)
    car.reset(pos=Vec3(0, 0, 0), yaw=0.0, initial_speed=10.0)
    for _ in range(60):
        car.step(steering_cmd=0.6, throttle_cmd=0.5, brake_cmd=0.0, dt=dt)

    assert car.state.yaw < -0.05, "Steer right should produce negative yaw angle"
    assert car.state.pos.y < -0.5, "Steer right should produce negative lateral displacement (-Y)"


def test_vehicle_obb_and_wheels():
    car = VehicleModel()
    car.reset(pos=Vec3(10.0, 5.0, 0.0), yaw=math.pi / 2.0)
    obb = car.get_obb()
    assert abs(obb.center.x - 10.0) < 1e-4
    assert abs(obb.center.y - 5.0) < 1e-4

    wheels = car.get_wheel_transforms()
    assert len(wheels) == 4
    for w_pos, steer in wheels:
        assert isinstance(w_pos, Vec3)


# ---------------------------------------------------------------- pacejka4

def test_pacejka4_normal_regime_matches_bicycle():
    """4-wheel Pacejka must agree with the axle model in linear regime."""
    results = {}
    for tm in ("bicycle", "pacejka4"):
        car = VehicleModel(VehicleConfig(tire_model=tm))
        car.reset(pos=Vec3(0, 0, 0), yaw=0.0, initial_speed=20.0)
        for _ in range(600):
            car.step(steering_cmd=-0.10, throttle_cmd=0.0, brake_cmd=0.0,
                     dt=1.0 / 60.0)
        results[tm] = car.state.yaw_rate
    assert abs(results["pacejka4"] - results["bicycle"]) < 0.05


def test_pacejka4_lateral_load_transfer():
    """Cornering must shift normal load to the outside wheels."""
    car = VehicleModel(VehicleConfig(tire_model="pacejka4"))
    car.reset(pos=Vec3(0, 0, 0), yaw=0.0, initial_speed=20.0)
    for _ in range(300):
        car.step(steering_cmd=-0.12, throttle_cmd=0.0, brake_cmd=0.0,
                 dt=1.0 / 60.0)
    fz = car.state.fz_wheels
    assert all(f > 0.0 for f in fz)
    assert abs(fz[0] - fz[1]) > 100.0 or abs(fz[2] - fz[3]) > 100.0
    assert abs(sum(fz) - 1200.0 * 9.81) < 500.0


def test_pacejka4_post_peak_decay():
    """Past peak slip the magic formula must shed force, not plateau."""
    car = VehicleModel(VehicleConfig(tire_model="pacejka4"))
    car.reset(pos=Vec3(0, 0, 0), yaw=0.0, initial_speed=20.0)
    g = 9.81
    fz_f = 1200 * g * 0.48
    fz_r = 1200 * g * 0.52
    forces = []
    for vy in (3.0, 6.0, 15.0, 22.0):
        car.state.alpha_w_eff = [0.0] * 4
        car.state.accel_body.x = 0.0
        for _ in range(400):
            car._pacejka4(20.0, vy, 0.0, 20.0, fz_f, fz_r, fz_f, fz_r,
                          1.0, 80000.0, 85000.0, 0.0, 0.0, 1 / 240, 20.0)
        forces.append(-car.state.fy_wheels[0])
    peak = max(forces)
    assert forces[-1] < peak * 0.995  # real decay, not a saturating cap

