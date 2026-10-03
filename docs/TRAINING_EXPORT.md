---
noteId: "35eaaa10bedc11f1a29f1fbaabbd87c8"
tags: []

---

# Training Export & Gymnasium Integration

## Overview

The **Training Exporter** (`sim_env/export.py`) bridges declarative visual authoring in the Studio with external Reinforcement Learning libraries (Stable-Baselines3, CleanRL, Ray RLlib, PyTorch).

It decouples the environment definition from external training code by exporting machine-readable JSON specifications and an auto-generated, standalone Gymnasium runner script.

---

## 1. Export Bundle Architecture

Executing `TrainingExporter.export_training_bundle()` outputs four coordinated files:

```
exported_training/
├── environment.json      # Complete declarative environment project (Track, Vehicle, Entities, Agent)
├── scenario.json         # Environmental situation overrides (Weather, Friction, Randomization)
├── experiment.json       # Training hyperparameters, algorithm budgets, evaluation cadence
└── run_gym_training.py   # Self-contained executable training runner
```

---

## 2. File Specifications

### `environment.json`
Contains the complete `EnvironmentProject` serialization (Schema 3.0.0):
- **Agent Definition**: Perceptual channels, action bounds, reward function, termination rules.
- **Track Definition**: Arc-length parameterized spline, boundary walls, curbs, spawn point.
- **Physics**: Vehicle kinematics, mass, steering rates, tire friction.
- **Integrity**: Deterministic SHA-256 fingerprint.

### `scenario.json`
Contains parametric environmental modifiers (`ScenarioDefinition`):
- Friction multiplier and weather condition.
- Target speed and time limit overrides.
- Domain randomization parameter ranges and seed.

### `experiment.json`
Specifies how the external algorithm interacts with the environment (`ExperimentConfig`):
- Algorithm type (e.g., `"PPO"`, `"SAC"`).
- Total timesteps and rollout steps.
- Learning rate, discount factor ($\gamma$), GAE coefficient ($\lambda$), clipping parameter ($\epsilon$).
- Evaluation frequency and checkpoint intervals.

### `run_gym_training.py`
A standalone, fully executable Python script that:
1. Loads `environment.json`, `scenario.json`, and `experiment.json`.
2. Instantiates `SimulationEnvironment` with zero GUI or renderer overhead.
3. Automatically sets up the compiled observation pipeline, action decoder, reward engine, and termination evaluator.
4. Executes the baseline training loop, logging returns, rollout metrics, and saving checkpoints.

---

## 3. Python Usage

### Programmatic Export

```python
from sim_env.export import TrainingExporter
from sim_project.serializer import EnvironmentProject

project = EnvironmentProject.load("my_track.sim.json")

exported_files = TrainingExporter.export_training_bundle(
    output_dir="experiments/lane_following_run",
    env_project=project,
    scenario=project.scenario_def,
    experiment=project.experiment_config
)
```

### Studio UI Export
In the Studio Editor mode:
1. Open the **OVERVIEW** inspector tab.
2. Click **EXPORT TRAINING**.
3. The training bundle is automatically generated in `experiments/exported_training/`.

---

## 4. Running External Training

The exported script can be run directly from the command line in headless environments or remote compute instances:

```bash
python experiments/exported_training/run_gym_training.py
```
