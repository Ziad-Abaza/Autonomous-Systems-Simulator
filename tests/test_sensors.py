"""
Unit tests for modular sensor suite, update scheduling, noise, and observations.
"""

import numpy as np
import pytest
from sim_core.math_utils import Vec3
from sim_core.sensors.sensor_manager import SensorManager
from sim_core.sensors.vehicle_state_sensor import VehicleStateSensor
from sim_core.sensors.raycast_sensor import RaycastSensor
from sim_core.sensors.imu_sensor import IMUSensor
from sim_core.sensors.camera_sensor import CameraSensor
from sim_core.vehicle.vehicle_model import VehicleModel
from sim_core.track.road_definition import RoadDefinition
from sim_core.track.mesh_generator import TrackMeshGenerator
from sim_core.track.track_queries import TrackSpatialQueries


@pytest.fixture
def test_context():
    road = RoadDefinition.create_default_oval()
    track = TrackMeshGenerator.generate(road)
    queries = TrackSpatialQueries(track)
    vehicle = VehicleModel()
    vehicle.reset(pos=Vec3(road.spawn_point.x, road.spawn_point.y, 0.0), yaw=road.spawn_point.yaw)
    return {
        'vehicle': vehicle,
        'track_queries': queries,
        'checkpoint_idx': 0,
        'obstacle_segments': [],
        'sim_time': 0.0,
    }


def test_vehicle_state_sensor(test_context):
    sensor = VehicleStateSensor(name="state", update_frequency_hz=60.0)
    rng = np.random.default_rng(42)
    sample = sensor.update(0.0, test_context, rng)

    assert 'speed' in sample
    assert 'heading_error' in sample
    assert 'distance_from_center' in sample
    assert 'is_on_road' in sample
    assert sample['is_on_road'] == 1.0


def test_raycast_sensor_beams(test_context):
    sensor = RaycastSensor(name="lidar", num_rays=15, fov_degrees=180.0, max_range=40.0)
    rng = np.random.default_rng(42)
    sample = sensor.update(0.0, test_context, rng)

    assert len(sample['distances']) == 15
    assert len(sample['ranges_norm']) == 15
    # Since car is inside track, rays hitting left/right boundaries should be <= 40.0
    assert np.all(sample['distances'] > 0.0)
    assert np.all(sample['ranges_norm'] <= 1.0)


def test_camera_sensor_image_generation(test_context):
    sensor = CameraSensor(name="cam", width=84, height=84)
    rng = np.random.default_rng(42)
    img = sensor.update(0.0, test_context, rng)

    assert isinstance(img, np.ndarray)
    assert img.shape == (84, 84, 3)
    assert img.dtype == np.uint8


def test_sensor_manager_scheduling(test_context):
    sm = SensorManager()
    s30 = RaycastSensor(name="ray30", update_frequency_hz=30.0)
    s60 = VehicleStateSensor(name="st60", update_frequency_hz=60.0)
    sm.add_sensor(s30)
    sm.add_sensor(s60)

    rng = np.random.default_rng(42)
    # At t=0.0: both update
    sm.update_all(0.0, test_context, rng)
    assert s30.get_last_sample() is not None
    assert s60.get_last_sample() is not None

    # At t = 1.0/60.0 (step 1 for 60Hz): s60 should update, s30 should not recompute
    dt_60 = 1.0 / 60.0
    assert s60.should_update(dt_60)
    assert not s30.should_update(dt_60)
