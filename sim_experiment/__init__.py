"""
Training & Experiment Platform (Phase 4).

Domain layer for reproducible RL experiments: immutable experiment manifests,
run lifecycle management, structured metrics, artifact registries, local
training orchestration, evaluation, trajectories, and batch expansion.

The simulation core (sim_core/sim_env) stays independent of this package:
the RL algorithm and training loop always live outside the simulator.
"""

from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.run import Run, RunStatus, RunManager
from sim_experiment.metrics import MetricsWriter, MetricsReader
from sim_experiment.artifacts import ArtifactRegistry

__all__ = [
    'ExperimentManifest',
    'TrainingConfig',
    'EvaluationConfig',
    'ExperimentManager',
    'Run',
    'RunStatus',
    'RunManager',
    'MetricsWriter',
    'MetricsReader',
    'ArtifactRegistry',
]
