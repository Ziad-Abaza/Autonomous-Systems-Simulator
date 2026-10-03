"""
Experiment Configuration Architecture.
Separates the physical/semantic Environment from the external RL training Experiment
(algorithms, training budgets, evaluation schedules, hyperparameters, and checkpoints).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class ExperimentConfig:
    """
    Defines how an RL algorithm interacts with an environment for training and evaluation.
    Keeps algorithm logic completely decoupled from the simulation core.
    """
    experiment_id: str = "exp_lane_keeping_01"
    name: str = "PPO Lane Keeping Baseline Experiment"
    environment_name: str = "Basic Driving Proving Ground"
    scenario_id: str = "basic_lane_following"
    seed: int = 42
    algorithm: str = "PPO"               # "PPO", "SAC", "DQN", "Custom"
    total_timesteps: int = 50000
    rollout_steps: int = 1024
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    batch_size: int = 256
    num_epochs: int = 4
    eval_frequency_steps: int = 5000
    eval_episodes: int = 5
    checkpoint_frequency_steps: int = 10000
    output_directory: str = "experiments/results"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        return cls(
            experiment_id=str(data.get("experiment_id", "exp_01")),
            name=str(data.get("name", "Experiment")),
            environment_name=str(data.get("environment_name", "Untitled Environment")),
            scenario_id=str(data.get("scenario_id", "default")),
            seed=int(data.get("seed", 42)),
            algorithm=str(data.get("algorithm", "PPO")),
            total_timesteps=int(data.get("total_timesteps", 50000)),
            rollout_steps=int(data.get("rollout_steps", 1024)),
            learning_rate=float(data.get("learning_rate", 3e-4)),
            gamma=float(data.get("gamma", 0.99)),
            gae_lambda=float(data.get("gae_lambda", 0.95)),
            clip_coef=float(data.get("clip_coef", 0.2)),
            batch_size=int(data.get("batch_size", 256)),
            num_epochs=int(data.get("num_epochs", 4)),
            eval_frequency_steps=int(data.get("eval_frequency_steps", 5000)),
            eval_episodes=int(data.get("eval_episodes", 5)),
            checkpoint_frequency_steps=int(data.get("checkpoint_frequency_steps", 10000)),
            output_directory=str(data.get("output_directory", "experiments/results")),
            hyperparameters=dict(data.get("hyperparameters", {}))
        )
