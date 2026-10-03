"""
Project serialization and persistence engine for environments, vehicle configs,
reward parameters, world entities, sensor suites, and scenarios into versioned *.sim.json files.
Supports Schema 2.0.0 with automatic backward migration for Schema 1.0.0 files.
"""

from __future__ import annotations
import json
import os
from typing import Dict, Any, Optional, List
from sim_core.track.road_definition import RoadDefinition
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.world.entity import WorldEntity, StaticObstacle, entity_from_dict
from sim_core.world.obstacle import Obstacle
from sim_env.spaces import ActionSpaceConfig, ObservationSchema
from sim_env.reward_engine import RewardConfig
from sim_env.termination_engine import TerminationConfig
from sim_env.domain_randomizer import DomainRandomizationConfig
from sim_env.scenarios import ScenarioConfig
SCHEMA_VERSION = "2.0.0"


class EnvironmentProject:
    """
    Serializable container representing a complete versioned AI environment.
    Schema 2.0.0 adds first-class World Entities, rich reward configurations,
    and modular sensor definitions.
    """
    SCHEMA_VERSION = "2.0.0"

    def __init__(
        self,
        name: str = "New Environment",
        road_def: Optional[RoadDefinition] = None,
        vehicle_config: Optional[VehicleConfig] = None,
        action_config: Optional[ActionSpaceConfig] = None,
        observation_schema: Optional[ObservationSchema] = None,
        reward_config: Optional[RewardConfig] = None,
        termination_config: Optional[TerminationConfig] = None,
        randomization_config: Optional[DomainRandomizationConfig] = None,
        scenario_config: Optional[ScenarioConfig] = None,
        entities: Optional[List[WorldEntity]] = None
    ):
        self.name = name
        self.road_def = road_def or RoadDefinition.create_default_oval()
        self.vehicle_config = vehicle_config or VehicleConfig()
        self.action_config = action_config or ActionSpaceConfig()
        self.observation_schema = observation_schema or ObservationSchema()
        self.reward_config = reward_config or RewardConfig()
        self.termination_config = termination_config or TerminationConfig()
        self.randomization_config = randomization_config or DomainRandomizationConfig()
        self.scenario_config = scenario_config or ScenarioConfig()
        self.entities: List[WorldEntity] = entities if entities is not None else []
        self.schema_version: str = self.SCHEMA_VERSION

    @property
    def obstacles(self) -> List[Any]:
        """Backward-compatibility accessor returning all obstacle entities."""
        return [e for e in self.entities if getattr(e, 'entity_type', None) in ('obstacle', 'static_obstacle') or isinstance(e, StaticObstacle)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': self.SCHEMA_VERSION,
            'name': self.name,
            'road_definition': self.road_def.to_dict(),
            'vehicle_config': self.vehicle_config.to_dict(),
            'action_config': self.action_config.to_dict(),
            'observation_schema': self.observation_schema.to_dict(),
            'reward_config': self.reward_config.to_dict(),
            'termination_config': self.termination_config.to_dict(),
            'randomization_config': self.randomization_config.to_dict(),
            'scenario_config': self.scenario_config.to_dict(),
            'entities': [e.to_dict() for e in self.entities],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentProject:
        version = str(data.get('schema_version', '1.0.0'))
        road_def = RoadDefinition.from_dict(data.get('road_definition', {}))
        vehicle_cfg = VehicleConfig.from_dict(data.get('vehicle_config', {}))
        action_cfg = ActionSpaceConfig.from_dict(data.get('action_config', {}))
        obs_schema = ObservationSchema.from_dict(data.get('observation_schema', {}))
        reward_cfg = RewardConfig.from_dict(data.get('reward_config', {}))
        term_cfg = TerminationConfig.from_dict(data.get('termination_config', {}))
        rand_cfg = DomainRandomizationConfig.from_dict(data.get('randomization_config', {}))
        scen_cfg = ScenarioConfig.from_dict(data.get('scenario_config', {}))

        # Entity deserialization with Schema 1.0.0 migration
        entities: List[WorldEntity] = []
        if 'entities' in data and data['entities']:
            for edata in data['entities']:
                try:
                    entities.append(entity_from_dict(edata))
                except Exception:
                    # Fallback generic obstacle
                    entities.append(StaticObstacle.from_dict(edata))
        elif 'obstacles' in data and data['obstacles']:
            for obs_data in data['obstacles']:
                try:
                    entities.append(Obstacle.from_dict(obs_data))
                except Exception:
                    pass
        elif scen_cfg.obstacles:
            # Backward migration: convert legacy scenario obstacles into WorldEntity
            for obs_data in scen_cfg.obstacles:
                try:
                    entities.append(Obstacle.from_dict(obs_data))
                except Exception:
                    pass

        return cls(
            name=str(data.get('name', 'Untitled Environment')),
            road_def=road_def,
            vehicle_config=vehicle_cfg,
            action_config=action_cfg,
            observation_schema=obs_schema,
            reward_config=reward_cfg,
            termination_config=term_cfg,
            randomization_config=rand_cfg,
            scenario_config=scen_cfg,
            entities=entities
        )

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> EnvironmentProject:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
