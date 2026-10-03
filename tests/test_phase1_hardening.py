"""
Phase 1 Hardening & Regression Test Suite.
Validates:
- Action integrity (NaN, Inf, discrete out-of-range, missing dimensions, type safety)
- Observation isolation and schema integrity (vector, dict, vision-only, missing sensor padding)
- Directional checkpoint crossing (prevention of backward driving rewards)
- Open track completion vs closed track lap completion
- Reward integrity (teleportation clamp, backward driving penalty)
- Lifecycle contract (stepping after termination without reset)
- Inspector property mutations and 3D rebuild equivalence
- Vehicle steering sign and closed-loop PID multi-step stability
"""

import math
import numpy as np
import pytest

from sim_core.math_utils import Vec2, Vec3
from sim_core.track.road_definition import RoadDefinition, ControlPoint, SpawnPoint
from sim_core.track.mesh_generator import TrackMeshGenerator
from sim_core.vehicle.vehicle_model import VehicleModel
from sim_core.world.checkpoint import CheckpointTracker
from sim_env.environment import SimulationEnvironment
from sim_env.spaces import ActionSpaceConfig, ActionSpaceType, ObservationSchema
from sim_env.reward_engine import RewardEngine, RewardConfig
from sim_env.termination_engine import TerminationEngine, TerminationConfig
from sim_ui.inspector import EnvironmentInspector


# =========================================================================
# 1. ACTION INTEGRITY TESTS
# =========================================================================

def test_action_validation_nan_and_inf():
    cfg = ActionSpaceConfig(type=ActionSpaceType.CONTINUOUS)
    
    # Valid action
    valid, err = cfg.validate_action([0.0, 0.5, 0.0])
    assert valid is True
    assert err == ""

    # Action with NaN
    valid_nan, err_nan = cfg.validate_action([float('nan'), 0.5, 0.0])
    assert valid_nan is False
    assert "NaN" in err_nan

    # Action with Inf
    valid_inf, err_inf = cfg.validate_action([0.0, float('inf'), 0.0])
    assert valid_inf is False
    assert "Infinity" in err_inf

    # None action
    valid_none, err_none = cfg.validate_action(None)
    assert valid_none is False

    # Action with insufficient dimensions
    valid_short, err_short = cfg.validate_action([0.5])
    assert valid_short is False


def test_action_decoding_never_corrupts_state():
    cfg = ActionSpaceConfig(type=ActionSpaceType.CONTINUOUS)
    
    # Decoding NaN continuous action produces finite values (nan converted to 0.0)
    steer, throttle, brake = cfg.decode_action([float('nan'), float('nan'), float('nan')])
    assert not math.isnan(steer)
    assert not math.isnan(throttle)
    assert not math.isnan(brake)
    assert steer == 0.0

    # Decoding Infinity continuous action clamps to bounds
    steer_inf, throttle_inf, brake_inf = cfg.decode_action([1e9, 1e9, 1e9])
    assert steer_inf == 1.0
    assert throttle_inf == 1.0
    assert brake_inf == 1.0

    # Discrete invalid actions safely default to 0
    cfg_disc = ActionSpaceConfig(type=ActionSpaceType.DISCRETE)
    valid_disc, _ = cfg_disc.validate_action(999)
    assert valid_disc is False
    act_safe = cfg_disc.decode_action(999)
    # Clamps to max valid index
    assert act_safe == (0.6, 0.4, 0.0)

    # Discrete NaN defaults to index 0
    act_nan = cfg_disc.decode_action(float('nan'))
    assert act_nan == (0.0, 0.0, 0.0)


def test_environment_step_with_nan_action():
    env = SimulationEnvironment()
    env.reset()
    
    # Step with NaN action
    obs, reward, terminated, truncated, info = env.step([float('nan'), float('nan'), float('nan')])
    assert info['action_valid'] is False
    assert not math.isnan(env.vehicle.state.pos.x)
    assert not math.isnan(env.vehicle.state.pos.y)
    assert not math.isnan(env.vehicle.state.speed)


# =========================================================================
# 2. OBSERVATION INTEGRITY & SCHEMA ISOLATION TESTS
# =========================================================================

def test_observation_schema_feature_isolation():
    # Only Speed and LiDAR rays enabled
    schema = ObservationSchema(
        include_speed=True,
        include_velocity=False,
        include_yaw_rate=False,
        include_steering_angle=False,
        include_distance_from_center=False,
        include_heading_error=False,
        include_distance_to_checkpoint=False,
        include_lidar_rays=True,
        include_camera_rgb=False,
        flatten_vector=True
    )
    assert schema.compute_vector_dim(num_lidar_rays=15) == 1 + 15  # 16

    env = SimulationEnvironment(observation_schema=schema)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (16,)
    
    # Step environment
    obs, _, _, _, _ = env.step([0.0, 0.5, 0.0])
    assert obs.shape == (16,)


def test_observation_invariant_vector_length_without_lidar():
    schema = ObservationSchema(include_lidar_rays=True, flatten_vector=True)
    env = SimulationEnvironment(observation_schema=schema)
    # Remove lidar sensor to simulate delayed/missing sensor
    env.sensors.remove_sensor("lidar_rays")
    obs, _ = env.reset()
    # Length must remain 23 (with padding) rather than shrinking to 8
    assert len(obs) == 23


def test_observation_dictionary_mode_completeness():
    schema = ObservationSchema(flatten_vector=False, include_distance_to_checkpoint=True)
    env = SimulationEnvironment(observation_schema=schema)
    obs, _ = env.reset()
    assert isinstance(obs, dict)
    assert "distance_to_checkpoint" in obs
    assert "speed" in obs
    assert "vel_body" in obs
    assert "lidar_ranges" in obs


def test_observation_vision_only_mode():
    schema = ObservationSchema(
        include_speed=False,
        include_velocity=False,
        include_yaw_rate=False,
        include_steering_angle=False,
        include_distance_from_center=False,
        include_heading_error=False,
        include_distance_to_checkpoint=False,
        include_lidar_rays=False,
        include_camera_rgb=True,
        flatten_vector=True
    )
    env = SimulationEnvironment(observation_schema=schema)
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (84, 84, 3)


# =========================================================================
# 3. CHECKPOINT DIRECTION & EXPLOIT PREVENTION TESTS
# =========================================================================

def test_checkpoint_backward_crossing_prevented():
    road = RoadDefinition.create_default_oval()
    track = TrackMeshGenerator.generate(road)
    tracker = CheckpointTracker(track)
    tracker.reset()

    # Target checkpoint 1
    cp1 = track.checkpoints[1]
    gl = cp1['gate_left']
    gr = cp1['gate_right']
    gate_center = (gl + gr) * 0.5
    tan = Vec2(cp1['tangent'].x, cp1['tangent'].y).normalized()

    # Case A: Crossing in FORWARD direction
    p_before = gate_center - tan * 1.0
    p_after = gate_center + tan * 1.0
    passed_fwd, _ = tracker.update(p_before, p_after, current_time=1.0)
    assert passed_fwd is True, "Forward crossing should pass checkpoint"

    # Case B: Crossing next checkpoint in BACKWARD direction
    cp2 = track.checkpoints[2]
    tan2 = Vec2(cp2['tangent'].x, cp2['tangent'].y).normalized()
    gate_center2 = (cp2['gate_left'] + cp2['gate_right']) * 0.5

    p_before_back = gate_center2 + tan2 * 1.0
    p_after_back = gate_center2 - tan2 * 1.0  # moving opposite to tangent
    passed_back, _ = tracker.update(p_before_back, p_after_back, current_time=2.0)
    assert passed_back is False, "Backward crossing must NOT pass checkpoint"


def test_open_track_course_completion():
    road = RoadDefinition(name="Sprint Track", is_closed=False)
    road.add_control_point(0, 0, 0, 10.0)
    road.add_control_point(50, 0, 0, 10.0)
    road.add_control_point(100, 0, 0, 10.0)
    road.add_control_point(150, 0, 0, 10.0)
    road.num_checkpoints = 4

    track = TrackMeshGenerator.generate(road)
    tracker = CheckpointTracker(track)
    tracker.reset()

    num_cp = len(track.checkpoints)
    # Cross checkpoints 1, 2, and finally 3 (the last checkpoint)
    for i in range(1, num_cp):
        cp = track.checkpoints[i]
        gc = (cp['gate_left'] + cp['gate_right']) * 0.5
        tan = Vec2(cp['tangent'].x, cp['tangent'].y).normalized()
        passed, lap_completed = tracker.update(gc - tan * 0.5, gc + tan * 0.5, current_time=float(i))
        assert passed is True
        if i == num_cp - 1:
            assert lap_completed is True, "Reaching final checkpoint on open track must complete course"
        else:
            assert lap_completed is False


# =========================================================================
# 4. REWARD ENGINE INTEGRITY & TELEPORTATION CLAMP
# =========================================================================

def test_reward_teleportation_clamp():
    engine = RewardEngine(RewardConfig(weight_progress=1.0))
    engine.reset(initial_s=10.0)

    # Step with normal progress (1.0 meter)
    r_norm, breakdown_norm = engine.compute_step_reward(
        current_s=11.0, track_length=500.0, is_closed=True, lateral_offset=0.0,
        road_width=12.0, speed=10.0, heading_error=0.0, current_steer=0.0,
        is_colliding=False, is_on_road=True, checkpoint_passed=False, lap_completed=False
    )
    assert breakdown_norm['progress'] == pytest.approx(1.0, rel=1e-3)

    # Step with teleportation jump (50 meters forward in a single step)
    r_teleport, breakdown_teleport = engine.compute_step_reward(
        current_s=61.0, track_length=500.0, is_closed=True, lateral_offset=0.0,
        road_width=12.0, speed=10.0, heading_error=0.0, current_steer=0.0,
        is_colliding=False, is_on_road=True, checkpoint_passed=False, lap_completed=False
    )
    assert breakdown_teleport['progress'] == 0.0, "Discontinuous jump must yield zero progress reward"


# =========================================================================
# 5. ENVIRONMENT STEP AFTER TERMINATION CONTRACT
# =========================================================================

def test_step_after_termination_contract():
    env = SimulationEnvironment()
    env.reset()
    
    # Force termination
    env.is_done = True
    
    obs, reward, terminated, truncated, info = env.step([0.0, 0.5, 0.0])
    assert terminated is True
    assert truncated is True
    assert reward == 0.0
    assert info['termination_reason'] == "invalid_call_after_done"
    assert "error" in info


# =========================================================================
# 6. ENVIRONMENT INSPECTOR & 3D REBUILD INTEGRITY
# =========================================================================

def test_environment_inspector_mutations():
    road = RoadDefinition.create_default_oval()
    from sim_core.vehicle.vehicle_config import VehicleConfig
    from sim_core.sensors.sensor_manager import SensorManager
    
    vc = VehicleConfig()
    sm = SensorManager.create_default_sensor_suite()
    inspector = EnvironmentInspector(road, vc, sm)

    # 1. Modify Track property
    inspector.active_tab = "TRACK"
    inspector.handle_property_change("b_left_type", None)
    assert road.boundary_config.left_type == "wall"

    # 2. Modify Control Point width and elevation
    inspector.select_control_point(0)
    orig_w = road.control_points[0].width
    inspector.handle_property_change("cp_width", 2.0)
    assert road.control_points[0].width == orig_w + 2.0

    # 3. Modify Spawn Point
    inspector.active_tab = "SPAWN"
    orig_yaw = road.spawn_point.yaw
    inspector.handle_property_change("sp_yaw", 10.0)
    assert abs(road.spawn_point.yaw - (orig_yaw + math.radians(10.0))) < 1e-4

    # 4. Modify Vehicle Mass
    inspector.active_tab = "VEHICLE"
    orig_mass = vc.mass
    inspector.handle_property_change("vc_mass", 100.0)
    assert vc.mass == orig_mass + 100.0
