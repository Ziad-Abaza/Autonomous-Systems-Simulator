"""
Experiment Manifest.

An immutable, self-describing record of exactly what an experiment is:
the full environment snapshot (not just a reference), the scenario,
the deterministic identity fingerprints, the algorithm-agnostic training
configuration, and the evaluation configuration.

Identity rules:
- environment_fingerprint comes from EnvironmentProject.compute_fingerprint()
  (structural, deterministic, excludes version/metadata fields).
- experiment_fingerprint is computed over the complete identity payload:
  environment fingerprint + scenario + schemas + episode/randomization/
  curriculum configuration + training configuration + seed + simulator
  and protocol versions. Human-readable names are metadata, not identity.
- experiment_id is a short, human-inspectable form of the fingerprint.
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from sim_version import SIMULATOR_VERSION
from sim_net.protocol import PROTOCOL_VERSION
from sim_env.versioning import EnvironmentVersionManager

MANIFEST_VERSION = "1.0"


def _canonical_hash(data: Dict[str, Any]) -> str:
    """Deterministic SHA-256 over a JSON-canonicalized dict."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class TrainingConfig:
    """
    Algorithm-agnostic training configuration.

    Common fields apply to any trainer; algorithm-specific hyperparameters
    live under algorithm_config so the experiment model is not coupled to PPO.
    """
    algorithm: str = "ppo"                       # "ppo", "sac", "dqn", "custom:<name>"
    total_timesteps: int = 50000
    rollout_length: int = 1024                   # steps per rollout / buffer segment
    batch_size: int = 256
    epochs: int = 4                              # optimization epochs per update (where applicable)
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95                     # where applicable
    eval_frequency: int = 5000                   # timesteps between evaluations (0 = off)
    checkpoint_frequency: int = 10000            # timesteps between checkpoints (0 = off)
    logging_frequency: int = 1                   # rollout updates between metric writes
    num_envs: int = 1                            # parallel environments (tcp mode)
    max_wall_seconds: float = 0.0                # 0 = unlimited run time
    algorithm_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        cfg = cls()
        for f_name in (
            "algorithm", "total_timesteps", "rollout_length", "batch_size",
            "epochs", "learning_rate", "discount_factor", "gae_lambda",
            "eval_frequency", "checkpoint_frequency", "logging_frequency",
            "num_envs", "max_wall_seconds",
        ):
            if f_name in data:
                setattr(cfg, f_name, data[f_name])
        cfg.algorithm_config = dict(data.get("algorithm_config", {}))
        return cfg


@dataclass
class EvaluationConfig:
    """Evaluation configuration, separate from training state."""
    eval_seeds: List[int] = field(default_factory=lambda: [0])
    num_episodes: int = 5
    deterministic_policy: bool = True
    scenario_id: Optional[str] = None            # None = training scenario

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        return cls(
            eval_seeds=list(data.get("eval_seeds", [0])),
            num_episodes=int(data.get("num_episodes", 5)),
            deterministic_policy=bool(data.get("deterministic_policy", True)),
            scenario_id=data.get("scenario_id"),
        )


@dataclass
class ExperimentManifest:
    """
    Immutable experiment definition. Once a run is launched the manifest
    must not be mutated — edits produce a new experiment identity.
    """
    experiment_id: str = ""
    manifest_version: str = MANIFEST_VERSION
    name: str = "experiment"
    created_at: float = 0.0                      # metadata only, not identity

    # Immutable environment identity + full snapshot
    environment_name: str = ""
    environment_version: str = "1.0.0"
    environment_fingerprint: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)

    # Scenario
    scenario_id: str = ""
    scenario_configuration: Dict[str, Any] = field(default_factory=dict)

    # Determinism + versions
    random_seed: int = 42
    simulator_version: str = SIMULATOR_VERSION
    protocol_version: str = PROTOCOL_VERSION

    # Contract schemas
    observation_schema: Dict[str, Any] = field(default_factory=dict)
    action_schema: Dict[str, Any] = field(default_factory=dict)
    reward_configuration: Dict[str, Any] = field(default_factory=dict)
    termination_configuration: Dict[str, Any] = field(default_factory=dict)
    episode_configuration: Dict[str, Any] = field(default_factory=dict)
    curriculum_configuration: Optional[Dict[str, Any]] = None
    randomization_configuration: Dict[str, Any] = field(default_factory=dict)

    # Training + evaluation
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Identity + lifecycle
    experiment_fingerprint: str = ""
    launched: bool = False                       # set by ExperimentManager on first run
    archived: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_project(
        cls,
        project: Any,                            # EnvironmentProject
        scenario: Any,                           # ScenarioDefinition
        training: Optional[TrainingConfig] = None,
        evaluation: Optional[EvaluationConfig] = None,
        name: str = "experiment",
        random_seed: int = 42,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ExperimentManifest":
        """Snapshots an EnvironmentProject into an immutable manifest."""
        env_dict = project.to_dict()
        env_fp = EnvironmentVersionManager.compute_fingerprint(env_dict)

        agent = getattr(project, "agent", None)
        obs_schema = agent.observation_space.export_schema() if agent else {}
        act_schema = agent.action_space.export_schema() if agent else {}
        reward_cfg = agent.reward_function.to_dict() if agent else {}
        term_cfg = agent.termination_rules.to_dict() if agent else {}

        m = cls(
            name=name,
            created_at=time.time(),
            environment_name=project.name,
            environment_version=project.environment_version,
            environment_fingerprint=env_fp,
            environment=env_dict,
            scenario_id=scenario.scenario_id if scenario else "",
            scenario_configuration=scenario.to_dict() if scenario else {},
            random_seed=int(random_seed),
            observation_schema=obs_schema,
            action_schema=act_schema,
            reward_configuration=reward_cfg,
            termination_configuration=term_cfg,
            episode_configuration=project.episode_config.to_dict() if getattr(project, "episode_config", None) else {},
            curriculum_configuration=project.curriculum.to_dict() if getattr(project, "curriculum", None) else None,
            randomization_configuration=scenario.randomization.to_dict() if scenario else {},
            training=training or TrainingConfig(),
            evaluation=evaluation or EvaluationConfig(),
            metadata=dict(metadata or {}),
        )
        m.experiment_fingerprint = m._compute_fingerprint()
        m.experiment_id = f"exp_{m.experiment_fingerprint[:16]}"
        return m

    def _identity_payload(self) -> Dict[str, Any]:
        """The exact fields that define experiment identity (no timestamps/names)."""
        return {
            "environment_fingerprint": self.environment_fingerprint,
            "scenario_configuration": self.scenario_configuration,
            "random_seed": self.random_seed,
            "simulator_version": self.simulator_version,
            "protocol_version": self.protocol_version,
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "reward_configuration": self.reward_configuration,
            "termination_configuration": self.termination_configuration,
            "episode_configuration": self.episode_configuration,
            "curriculum_configuration": self.curriculum_configuration,
            "randomization_configuration": self.randomization_configuration,
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }

    def _compute_fingerprint(self) -> str:
        return _canonical_hash(self._identity_payload())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "experiment_id": self.experiment_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "name": self.name,
            "created_at": self.created_at,
            "environment_name": self.environment_name,
            "environment_version": self.environment_version,
            "environment_fingerprint": self.environment_fingerprint,
            "environment": self.environment,
            "scenario_id": self.scenario_id,
            "scenario_configuration": self.scenario_configuration,
            "random_seed": self.random_seed,
            "simulator_version": self.simulator_version,
            "protocol_version": self.protocol_version,
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "reward_configuration": self.reward_configuration,
            "termination_configuration": self.termination_configuration,
            "episode_configuration": self.episode_configuration,
            "curriculum_configuration": self.curriculum_configuration,
            "randomization_configuration": self.randomization_configuration,
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "launched": self.launched,
            "archived": self.archived,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentManifest":
        m = cls(
            manifest_version=str(data.get("manifest_version", MANIFEST_VERSION)),
            experiment_id=str(data.get("experiment_id", "")),
            name=str(data.get("name", "experiment")),
            created_at=float(data.get("created_at", 0.0)),
            environment_name=str(data.get("environment_name", "")),
            environment_version=str(data.get("environment_version", "1.0.0")),
            environment_fingerprint=str(data.get("environment_fingerprint", "")),
            environment=dict(data.get("environment", {})),
            scenario_id=str(data.get("scenario_id", "")),
            scenario_configuration=dict(data.get("scenario_configuration", {})),
            random_seed=int(data.get("random_seed", 42)),
            simulator_version=str(data.get("simulator_version", SIMULATOR_VERSION)),
            protocol_version=str(data.get("protocol_version", PROTOCOL_VERSION)),
            observation_schema=dict(data.get("observation_schema", {})),
            action_schema=dict(data.get("action_schema", {})),
            reward_configuration=dict(data.get("reward_configuration", {})),
            termination_configuration=dict(data.get("termination_configuration", {})),
            episode_configuration=dict(data.get("episode_configuration", {})),
            curriculum_configuration=data.get("curriculum_configuration"),
            randomization_configuration=dict(data.get("randomization_configuration", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation", {})),
            experiment_fingerprint=str(data.get("experiment_fingerprint", "")),
            launched=bool(data.get("launched", False)),
            archived=bool(data.get("archived", False)),
            metadata=dict(data.get("metadata", {})),
        )
        return m
