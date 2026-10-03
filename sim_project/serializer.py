"""
Project serialization and persistence engine for environments, vehicle configs,
reward parameters, world entities, sensor suites, and scenarios into versioned *.sim.json files.
Supports Schema 2.0.0 and Schema 3.0.0 with automatic migration from Schema 1.0.0 and 2.0.0.
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
from sim_env.agent import AgentDefinition
from sim_env.episode_config import EpisodeConfiguration
from sim_env.scenario_designer import ScenarioDefinition
from sim_env.curriculum import CurriculumDefinition
from sim_env.experiment import ExperimentConfig
from sim_env.versioning import EnvironmentVersionManager, EnvironmentVersion

SCHEMA_VERSION = "2.0.0"
SCHEMA_VERSION_V3 = "3.0.0"


class EnvironmentProject:
    """
    Serializable container representing a complete versioned AI environment.
    Supports declarative Agents, composable Observation/Action/Reward/Termination designers,
    curriculum learning, and reproducible domain randomization.
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
        entities: Optional[List[WorldEntity]] = None,
        agent: Optional[AgentDefinition] = None,
        episode_config: Optional[EpisodeConfiguration] = None,
        scenario_def: Optional[ScenarioDefinition] = None,
        curriculum: Optional[CurriculumDefinition] = None,
        experiment_config: Optional[ExperimentConfig] = None,
        environment_version: str = "1.0.0",
        schema_version: str = "2.0.0"
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
        self.schema_version: str = schema_version
        self.environment_version: str = environment_version

        # Phase 3 Declarative Subsystems
        self.agent: AgentDefinition = agent or AgentDefinition.create_default_vehicle_agent()
        self.episode_config: EpisodeConfiguration = episode_config or EpisodeConfiguration()
        self.scenario_def: ScenarioDefinition = scenario_def or ScenarioDefinition.get_standard_scenarios()["basic_lane_following"]
        self.curriculum: Optional[CurriculumDefinition] = curriculum
        self.experiment_config: Optional[ExperimentConfig] = experiment_config
        self._last_saved_fingerprint: Optional[str] = None

    @property
    def obstacles(self) -> List[Any]:
        """Backward-compatibility accessor returning all obstacle entities."""
        return [
            e for e in self.entities
            if getattr(e, 'entity_type', None) in ('obstacle', 'static_obstacle') or isinstance(e, StaticObstacle)
        ]

    def compute_fingerprint(self) -> str:
        """Computes deterministic SHA-256 fingerprint of current configuration."""
        return EnvironmentVersionManager.compute_fingerprint(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'schema_version': self.schema_version,
            'environment_version': self.environment_version,
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
            'agent': self.agent.to_dict(),
            'episode_config': self.episode_config.to_dict(),
            'scenario_def': self.scenario_def.to_dict(),
        }
        if self.curriculum is not None:
            data['curriculum'] = self.curriculum.to_dict()
        if self.experiment_config is not None:
            data['experiment_config'] = self.experiment_config.to_dict()

        data['fingerprint'] = EnvironmentVersionManager.compute_fingerprint(data)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentProject:
        version = str(data.get('schema_version', '1.0.0'))
        env_ver = str(data.get('environment_version', '1.0.0'))
        road_def = RoadDefinition.from_dict(data.get('road_definition', {}))
        vehicle_cfg = VehicleConfig.from_dict(data.get('vehicle_config', {}))
        action_cfg = ActionSpaceConfig.from_dict(data.get('action_config', {}))
        obs_schema = ObservationSchema.from_dict(data.get('observation_schema', {}))
        reward_cfg = RewardConfig.from_dict(data.get('reward_config', {}))
        term_cfg = TerminationConfig.from_dict(data.get('termination_config', {}))
        rand_cfg = DomainRandomizationConfig.from_dict(data.get('randomization_config', {}))
        scen_cfg = ScenarioConfig.from_dict(data.get('scenario_config', {}))

        # Entity deserialization with Schema 1.0.0 and 2.0.0 migration
        entities: List[WorldEntity] = []
        if 'entities' in data and data['entities']:
            for edata in data['entities']:
                try:
                    entities.append(entity_from_dict(edata))
                except Exception:
                    entities.append(StaticObstacle.from_dict(edata))
        elif 'obstacles' in data and data['obstacles']:
            for obs_data in data['obstacles']:
                try:
                    entities.append(Obstacle.from_dict(obs_data))
                except Exception:
                    pass
        elif scen_cfg.obstacles:
            for obs_data in scen_cfg.obstacles:
                try:
                    entities.append(Obstacle.from_dict(obs_data))
                except Exception:
                    pass

        # Phase 3 Agent
        if 'agent' in data and data['agent']:
            agent = AgentDefinition.from_dict(data['agent'])
        else:
            agent = AgentDefinition.create_default_vehicle_agent()

        # Phase 3 Episode Config
        if 'episode_config' in data and data['episode_config']:
            episode_cfg = EpisodeConfiguration.from_dict(data['episode_config'])
        else:
            episode_cfg = EpisodeConfiguration(
                max_steps=term_cfg.max_episode_steps,
                max_duration_seconds=float(term_cfg.max_episode_steps) / 60.0,
                initial_speed=road_def.spawn_point.initial_speed
            )

        # Phase 3 Scenario Def
        if 'scenario_def' in data and data['scenario_def']:
            scen_def = ScenarioDefinition.from_dict(data['scenario_def'])
        else:
            scen_def = ScenarioDefinition.get_standard_scenarios()["basic_lane_following"]

        curriculum = CurriculumDefinition.from_dict(data['curriculum']) if 'curriculum' in data and data['curriculum'] else None
        experiment = ExperimentConfig.from_dict(data['experiment_config']) if 'experiment_config' in data and data['experiment_config'] else None

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
            entities=entities,
            agent=agent,
            episode_config=episode_cfg,
            scenario_def=scen_def,
            curriculum=curriculum,
            experiment_config=experiment,
            environment_version=env_ver,
            schema_version=version if version in ("2.0.0", "3.0.0") else SCHEMA_VERSION
        )

    def save(self, filepath: str, auto_increment: bool = True) -> None:
        """Saves project to JSON file, automatically incrementing patch version if configuration changed."""
        if auto_increment and hasattr(self, '_last_saved_fingerprint'):
            curr_fp = self.compute_fingerprint()
            if self._last_saved_fingerprint and curr_fp != self._last_saved_fingerprint:
                v = EnvironmentVersion.from_string(self.environment_version)
                v.increment_patch()
                self.environment_version = v.to_string()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        self._last_saved_fingerprint = payload.get('fingerprint', self.compute_fingerprint())

    @classmethod
    def load(cls, filepath: str) -> EnvironmentProject:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        proj = cls.from_dict(data)
        proj._last_saved_fingerprint = proj.compute_fingerprint()
        return proj
