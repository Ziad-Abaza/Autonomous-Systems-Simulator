"""EnvFactory — TrackSpec + presets -> configured SimulationEnvironment.

Injects the agentRL observation subset, reward preset, termination
preset, and minimal sensor suite into a clone of the project's
declarative AgentDefinition, then materializes via
`sim_experiment.headless.build_env_from_dicts` (the canonical factory —
no sim internals modified, no duplicated construction logic).
"""
from __future__ import annotations

import copy
from typing import Any

from sim_env.agent import AgentDefinition
from sim_env.observation_designer import ObservationSpaceDefinition
from sim_env.sensor_config import default_suite_configs
from sim_experiment.headless import build_env_from_dicts
from sim_env.environment import SimulationEnvironment

from agentRL.envs.track_registry import TrackSpec
from agentRL.obs.spec import ObservationSpec, PRESETS
from agentRL.rewards.presets import reward_preset, termination_preset

# channel -> source sensor required in the suite
_CHANNEL_SENSORS = {
    "speed": "vehicle_state", "velocity_body": "vehicle_state",
    "yaw_rate": "vehicle_state", "steering_angle": "vehicle_state",
    "distance_from_center": "vehicle_state", "heading_error": "vehicle_state",
    "distance_to_checkpoint": "vehicle_state",
    "lidar_ranges": "lidar_rays",
}


class EnvFactory:
    """Builds SimulationEnvironment instances with the agentRL contract."""

    def __init__(
        self,
        reward: str = "drive_v1",
        termination: str = "term_v1",
        obs_spec: ObservationSpec | None = None,
        sensor_names: tuple[str, ...] | None = None,
        max_duration_s: float | None = 120.0,
        initial_speed: float | None = None,
    ) -> None:
        self.reward_name = reward
        self.termination_name = termination
        self.obs_spec = obs_spec or ObservationSpec(
            channel_names=PRESETS["full23"])
        self.sensor_names = sensor_names
        self.max_duration_s = max_duration_s
        self.initial_speed = initial_speed

    # -- project-dict surgery -------------------------------------------------

    def _needed_sensors(self) -> set[str]:
        if self.sensor_names is not None:
            return set(self.sensor_names)
        needed = {_CHANNEL_SENSORS[c] for c in self.obs_spec.channel_names}
        if self.obs_spec.image:
            needed.add("rgb_camera")
        return needed

    def _obs_space_dict(self) -> dict[str, Any]:
        base = ObservationSpaceDefinition.create_default_space()
        by_name = {c.name: c for c in base.channels}
        base.channels = [by_name[n] for n in self.obs_spec.channel_names]
        return base.to_dict()

    def project_dict(self, track: TrackSpec) -> dict[str, Any]:
        """Clone track.project with the agentRL contract injected."""
        proj = copy.deepcopy(track.project)
        agent = proj.get("agent") or AgentDefinition \
            .create_default_vehicle_agent().to_dict()
        agent["observation_space"] = self._obs_space_dict()
        agent["reward_function"] = reward_preset(
            self.reward_name).to_dict()
        agent["termination_rules"] = termination_preset(
            self.termination_name).to_dict()
        needed = self._needed_sensors()
        sensor_cfgs = [
            s.to_dict() for s in default_suite_configs()
            if s.name in needed
        ]
        for cfg in sensor_cfgs:
            cfg["enabled"] = True
        agent["sensor_configs"] = sensor_cfgs
        agent["sensor_names"] = [c["name"] for c in sensor_cfgs]
        proj["agent"] = agent
        if self.max_duration_s is not None:
            proj.setdefault("episode_config", {})
            proj["episode_config"]["max_duration_seconds"] = \
                float(self.max_duration_s)
        if self.initial_speed is not None:
            proj.setdefault("road_definition", {}).setdefault(
                "spawn_point", {})
            proj["road_definition"]["spawn_point"]["initial_speed"] = \
                float(self.initial_speed)
            proj["episode_config"]["initial_speed"] = \
                float(self.initial_speed)
        return proj

    def with_obs_spec(self, spec: ObservationSpec) -> "EnvFactory":
        """Clone with a different observation spec (e.g. an agent's own)."""
        return EnvFactory(reward=self.reward_name,
                          termination=self.termination_name,
                          obs_spec=spec,
                          sensor_names=self.sensor_names,
                          max_duration_s=self.max_duration_s,
                          initial_speed=self.initial_speed)

    def build(self, track: TrackSpec,
              seed: int = 42) -> SimulationEnvironment:
        return build_env_from_dicts(self.project_dict(track), seed=seed)
