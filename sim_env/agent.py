"""
Conceptual Agent Definition.
Decouples agent perception, action, rewards, and objectives from concrete
entity implementations. An Agent controls a WorldEntity (e.g. VehicleModel).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from sim_env.observation_designer import ObservationSpaceDefinition
from sim_env.action_designer import ActionSpaceDefinition
from sim_env.reward_designer import RewardFunctionDefinition
from sim_env.termination_designer import TerminationDefinition
from sim_env.sensor_config import SensorConfig, default_suite_configs, \
    configs_from_legacy_names


@dataclass
class AgentSpawnConfig:
    """Agent spawn pose and kinematic initialization."""
    spawn_mode: str = "default"  # "default", "custom", "checkpoint"
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.2)
    yaw_deg: float = 0.0
    initial_speed: float = 0.0
    lateral_jitter_m: float = 0.0
    heading_jitter_deg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spawn_mode": self.spawn_mode,
            "pos": list(self.pos),
            "yaw_deg": self.yaw_deg,
            "initial_speed": self.initial_speed,
            "lateral_jitter_m": self.lateral_jitter_m,
            "heading_jitter_deg": self.heading_jitter_deg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentSpawnConfig:
        return cls(
            spawn_mode=str(data.get("spawn_mode", "default")),
            pos=tuple(data.get("pos", [0.0, 0.0, 0.2])),
            yaw_deg=float(data.get("yaw_deg", 0.0)),
            initial_speed=float(data.get("initial_speed", 0.0)),
            lateral_jitter_m=float(data.get("lateral_jitter_m", 0.0)),
            heading_jitter_deg=float(data.get("heading_jitter_deg", 0.0))
        )


@dataclass
class AgentDefinition:
    """
    Conceptual Agent definition.
    Defines who the agent is, what entity it commands, what it observes,
    how it acts, how its objective is scored, and how its episodes terminate.
    """
    agent_id: str = "agent_01"
    name: str = "Autonomous Vehicle Agent"
    entity_type: str = "vehicle"          # Extensible to other domains (e.g. "drone", "robot")
    entity_id: Optional[str] = "vehicle_01"
    # Full declarative sensor suite — the serialized source of truth.
    # `sensor_names` is kept as a derived, backward-compatible name list.
    sensor_configs: List[SensorConfig] = field(default_factory=default_suite_configs)
    sensor_names: List[str] = field(default_factory=lambda: ["vehicle_state", "lidar_rays", "rgb_camera", "imu"])
    observation_space: ObservationSpaceDefinition = field(default_factory=ObservationSpaceDefinition.create_default_space)
    action_space: ActionSpaceDefinition = field(default_factory=ActionSpaceDefinition.create_default_vehicle_action_space)
    reward_function: RewardFunctionDefinition = field(default_factory=RewardFunctionDefinition.create_default_racing_reward)
    termination_rules: TerminationDefinition = field(default_factory=TerminationDefinition.create_default_racing_termination)
    spawn_config: AgentSpawnConfig = field(default_factory=AgentSpawnConfig)

    @classmethod
    def create_default_vehicle_agent(cls, agent_id: str = "vehicle_agent_01") -> AgentDefinition:
        """Standard autonomous racing vehicle agent implementation."""
        return cls(
            agent_id=agent_id,
            name="Autonomous Racing Vehicle",
            entity_type="vehicle",
            entity_id="vehicle_01",
            sensor_configs=default_suite_configs(),
            sensor_names=["vehicle_state", "lidar_rays", "rgb_camera", "imu"],
            observation_space=ObservationSpaceDefinition.create_default_space(),
            action_space=ActionSpaceDefinition.create_default_vehicle_action_space(continuous=True),
            reward_function=RewardFunctionDefinition.create_default_racing_reward(),
            termination_rules=TerminationDefinition.create_default_racing_termination(),
            spawn_config=AgentSpawnConfig(spawn_mode="default")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "sensor_configs": [s.to_dict() for s in self.sensor_configs],
            "sensor_names": [s.name for s in self.sensor_configs if s.enabled],
            "observation_space": self.observation_space.to_dict(),
            "action_space": self.action_space.to_dict(),
            "reward_function": self.reward_function.to_dict(),
            "termination_rules": self.termination_rules.to_dict(),
            "spawn_config": self.spawn_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentDefinition:
        obs_space = ObservationSpaceDefinition.from_dict(data.get("observation_space", {}))
        act_space = ActionSpaceDefinition.from_dict(data.get("action_space", {}))
        reward_fn = RewardFunctionDefinition.from_dict(data.get("reward_function", {}))
        term_rules = TerminationDefinition.from_dict(data.get("termination_rules", {}))
        spawn_cfg = AgentSpawnConfig.from_dict(data.get("spawn_config", {}))

        # Sensor suite: prefer the full declarative configs; fall back to
        # the legacy name list for files written before sensor authoring.
        raw_cfg = data.get("sensor_configs")
        if raw_cfg:
            sensor_configs = [SensorConfig.from_dict(s) for s in raw_cfg]
        else:
            sensor_configs = configs_from_legacy_names(list(
                data.get("sensor_names",
                         ["vehicle_state", "lidar_rays", "rgb_camera", "imu"])))
        return cls(
            agent_id=str(data.get("agent_id", "agent_01")),
            name=str(data.get("name", "Autonomous Vehicle Agent")),
            entity_type=str(data.get("entity_type", "vehicle")),
            entity_id=data.get("entity_id", "vehicle_01"),
            sensor_configs=sensor_configs,
            sensor_names=[s.name for s in sensor_configs if s.enabled],
            observation_space=obs_space,
            action_space=act_space,
            reward_function=reward_fn,
            termination_rules=term_rules,
            spawn_config=spawn_cfg
        )
