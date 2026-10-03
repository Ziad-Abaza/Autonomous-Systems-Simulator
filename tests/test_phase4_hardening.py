"""
Phase 3 hardening regression tests required for Phase 4.

Covers the gaps still open after the Phase 3 audit:
- ScenarioDefinition.time_limit_override consumed at runtime (truncation)
- ScenarioDefinition.sensor_noise_mult and randomized sensor_noise_mult consumed at runtime
- Environment fingerprint excludes version/metadata fields
- TCP protocol version negotiation
- Recorder captures simulator/protocol version metadata
"""

import numpy as np
import pytest

from sim_env import SimulationEnvironment
from sim_env.scenario_designer import ScenarioDefinition
from sim_env.randomization_designer import DomainRandomizationDefinition, RandomParamConfig, DistributionType
from sim_env.episode_config import EpisodeConfiguration
from sim_project.serializer import EnvironmentProject
from sim_net.server import SimulationServer
from sim_net.protocol import MessageType, PROTOCOL_VERSION
from sim_recorder.recorder import EpisodeRecorder
from sim_core.sensors.raycast_sensor import RaycastSensor
from sim_core.sensors.sensor_manager import SensorManager
from sim_core.sensors.vehicle_state_sensor import VehicleStateSensor


def _base_scenario(**kwargs) -> ScenarioDefinition:
    return ScenarioDefinition(scenario_id="test_scen", name="Test Scenario", **kwargs)


class TestScenarioTimeLimitOverride:
    def test_time_limit_override_truncates_episode(self):
        scen = _base_scenario(time_limit_override=0.1)  # ~6 steps at 60 Hz
        ep_cfg = EpisodeConfiguration(max_duration_seconds=60.0, max_steps=3600)
        env = SimulationEnvironment(scenario_def=scen, episode_config=ep_cfg, seed=7)
        env.reset(seed=7)

        term, trunc, reason = False, False, ""
        for _ in range(30):
            _, _, term, trunc, info = env.step([0.0, 0.3, 0.0])
            reason = info["termination_reason"]
            if term or trunc:
                break

        assert trunc is True, f"Expected truncation by scenario time limit, got reason={reason}"
        assert "time_limit" in reason or "duration" in reason

    def test_no_override_uses_episode_duration(self):
        scen = _base_scenario()
        ep_cfg = EpisodeConfiguration(max_duration_seconds=0.1, max_steps=3600)
        env = SimulationEnvironment(scenario_def=scen, episode_config=ep_cfg, seed=7)
        env.reset(seed=7)

        truncated = False
        for _ in range(30):
            _, _, term, trunc, info = env.step([0.0, 0.3, 0.0])
            if trunc:
                truncated = True
                assert info["termination_reason"] == "max_duration_exceeded"
                break
            assert not term, "Episode should truncate by duration, not terminate"
        assert truncated


class TestScenarioSensorNoiseMult:
    def _env_with_noisy_lidar(self, scen):
        sm = SensorManager()
        sm.add_sensor(VehicleStateSensor(name="vehicle_state", update_frequency_hz=60.0))
        sm.add_sensor(RaycastSensor(
            name="lidar_rays", num_rays=15, fov_degrees=180.0, max_range=40.0,
            update_frequency_hz=30.0, noise_std=0.05
        ))
        return SimulationEnvironment(sensor_manager=sm, scenario_def=scen, seed=3)

    def test_sensor_noise_mult_scales_base_noise(self):
        scen = _base_scenario(sensor_noise_mult=2.0)
        env = self._env_with_noisy_lidar(scen)
        env.reset(seed=3)
        lidar = env.sensors.get_sensor("lidar_rays")
        assert abs(lidar.noise_std - 0.10) < 1e-9

    def test_sensor_noise_mult_is_idempotent_across_resets(self):
        scen = _base_scenario(sensor_noise_mult=2.0)
        env = self._env_with_noisy_lidar(scen)
        env.reset(seed=3)
        env.reset(seed=3)
        lidar = env.sensors.get_sensor("lidar_rays")
        assert abs(lidar.noise_std - 0.10) < 1e-9  # not compounded to 0.2

    def test_randomization_sensor_noise_mult_stacks(self):
        rand = DomainRandomizationDefinition(
            enabled=True,
            global_seed=5,
            parameters={
                "sensor_noise_mult": RandomParamConfig(
                    param_name="sensor_noise_mult",
                    distribution=DistributionType.FIXED,
                    param1=1.5
                )
            }
        )
        scen = _base_scenario(sensor_noise_mult=2.0, randomization=rand)
        env = self._env_with_noisy_lidar(scen)
        env.reset(seed=3)
        lidar = env.sensors.get_sensor("lidar_rays")
        # 0.05 base * 2.0 scenario * 1.5 randomization
        assert abs(lidar.noise_std - 0.15) < 1e-9


class TestFingerprintStability:
    def test_version_fields_do_not_change_fingerprint(self):
        proj = EnvironmentProject(name="FP Test")
        fp_a = proj.compute_fingerprint()
        proj.environment_version = "9.9.9"
        fp_b = proj.compute_fingerprint()
        assert fp_a == fp_b, "environment_version bump must not change structural fingerprint"

    def test_config_change_changes_fingerprint(self):
        proj = EnvironmentProject(name="FP Test")
        fp_a = proj.compute_fingerprint()
        proj.agent.reward_function.components[0].weight += 1.0
        fp_b = proj.compute_fingerprint()
        assert fp_a != fp_b


class TestProtocolNegotiation:
    def _server(self):
        return SimulationServer(SimulationEnvironment(), host="127.0.0.1", port=0)

    def test_handshake_compatible_version(self):
        server = self._server()
        msg_type, payload = server._handle_message(
            MessageType.HANDSHAKE, {"protocol_versions": [PROTOCOL_VERSION]}
        )
        assert msg_type == MessageType.HANDSHAKE_ACK
        assert payload["protocol_version"] == PROTOCOL_VERSION
        assert PROTOCOL_VERSION in payload["supported_versions"]

    def test_handshake_incompatible_version_rejected(self):
        server = self._server()
        msg_type, payload = server._handle_message(
            MessageType.HANDSHAKE, {"protocol_versions": ["9.9"]}
        )
        assert msg_type == MessageType.ERROR
        assert "protocol" in payload["error"].lower()
        assert PROTOCOL_VERSION in payload.get("supported_versions", [])

    def test_handshake_no_version_declared_is_backward_compatible(self):
        server = self._server()
        msg_type, payload = server._handle_message(MessageType.HANDSHAKE, {})
        assert msg_type == MessageType.HANDSHAKE_ACK
        assert payload["protocol_version"] == PROTOCOL_VERSION


class TestRecorderMetadata:
    def test_recorder_stores_simulator_and_protocol_version(self):
        rec = EpisodeRecorder()
        rec.start_recording(track_name="t", seed=1)
        assert rec.metadata["simulator_version"]
        assert rec.metadata["protocol_version"] == PROTOCOL_VERSION

    def test_recorder_stores_scenario_configuration(self):
        rec = EpisodeRecorder()
        scen = _base_scenario(target_speed_override=11.0)
        rec.start_recording(
            track_name="t", seed=1,
            scenario_name=scen.name,
            scenario_config=scen.to_dict()
        )
        assert rec.metadata["scenario_config"]["target_speed_override"] == 11.0
