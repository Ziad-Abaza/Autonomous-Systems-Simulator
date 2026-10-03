---
noteId: "14326bf0bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Experiment Manifest

**File:** `sim_experiment/manifest.py` — persisted as
`experiments/<experiment_id>/experiment.json`

An experiment is the **immutable definition of a training task**. Once any
run launches, the manifest cannot change — modifications produce a new
experiment identity.

## Fields

| Field | Role |
|-------|------|
| `manifest_version` | `"1.0"` — versioned format for migration |
| `experiment_id` | `exp_<16 hex>` — short form of the fingerprint |
| `experiment_fingerprint` | SHA-256 of the identity payload |
| `name` | Human label — **metadata, not identity** |
| `created_at` | Timestamp — metadata, excluded from identity |
| `environment` | **Full serialized environment snapshot** (not a reference) |
| `environment_name` / `environment_version` / `environment_fingerprint` | Env identity preserved independently |
| `scenario_id` / `scenario_configuration` | Scenario + full scenario dict incl. randomization |
| `random_seed` | Base seed flowing to env + trainer |
| `simulator_version` | From `sim_version.SIMULATOR_VERSION` (`4.0.0`) |
| `protocol_version` | Negotiated TCP protocol (`2.0`) |
| `observation_schema` / `action_schema` | Exported agent contracts |
| `reward_configuration` / `termination_configuration` | Phase 3 agent definitions |
| `episode_configuration` | `EpisodeConfiguration` dict |
| `curriculum_configuration` | Optional curriculum dict |
| `randomization_configuration` | Domain randomization dict |
| `training` | `TrainingConfig` |
| `evaluation` | `EvaluationConfig` |
| `launched` / `archived` | Lifecycle flags (managed by `ExperimentManager`) |
| `metadata` | Free-form extension map |

## TrainingConfig (algorithm-agnostic)

Common fields apply to any trainer; algorithm-specific hyperparameters
live under `algorithm_config`:

```python
TrainingConfig(
    algorithm="ppo",          # or "sac", "dqn", "custom:<name>"
    total_timesteps=50000,
    rollout_length=1024,
    batch_size=256,
    epochs=4,
    learning_rate=3e-4,
    discount_factor=0.99,
    gae_lambda=0.95,
    eval_frequency=5000,      # 0 = off
    checkpoint_frequency=10000,
    logging_frequency=1,
    num_envs=1,               # parallel environments
    max_wall_seconds=0.0,     # 0 = unlimited
    algorithm_config={"clip_coef": 0.2, "ent_coef": 0.01, ...},
)
```

## EvaluationConfig

Evaluation is configured separately from training:

```python
EvaluationConfig(
    eval_seeds=[0, 1, 2],
    num_episodes=5,
    deterministic_policy=True,
    scenario_id=None,          # None = training scenario
)
```

## Immutability rules

- `ExperimentManager.create()` writes `experiment.json`,
  `environment.json`, `scenario.json`.
- `mark_launched()` sets `launched: true` — after that, `create()` and
  `save()` for the same id raise `RuntimeError`.
- `archive()` and run bookkeeping remain legal (lifecycle metadata only).

## Creation

```python
manifest = ExperimentManifest.from_project(
    project=project,                    # EnvironmentProject snapshot
    scenario=project.scenario_def,
    training=TrainingConfig(algorithm="ppo", total_timesteps=50000),
    evaluation=EvaluationConfig(eval_seeds=[0,1], num_episodes=3),
    name="lane-keep baseline",
    random_seed=42,
)
exp_dir = ExperimentManager("experiments").create(manifest)
```
