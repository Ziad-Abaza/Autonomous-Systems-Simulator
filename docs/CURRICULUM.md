---
noteId: "f52fd6c0bf0911f1a29f1fbaabbd87c8"
tags: []

---

# Curriculum Architecture & Runtime

## Model

`sim_env.curriculum` is the **declarative** model — a `CurriculumDefinition`
is an ordered list of `CurriculumStage`s:

| field | meaning |
|---|---|
| `scenario_id` | scenario the stage trains/evaluates against (standard library or the experiment's own scenario) |
| `target_metric` | evaluation metric that gates advancement |
| `advancement_threshold` | `>=` for maximize metrics, `<=` for lower-is-better (collision_rate, off_road_rate, timeout_rate, error metrics) |
| `min_episodes` | minimum *training* episodes inside the stage before advancement is allowed |
| `environment_overrides` | mapped onto `ScenarioDefinition` fields: `target_speed`→`target_speed_override`, `time_limit`→`time_limit_override`, `surface_friction_mult`, `sensor_noise_mult`, `ambient_light`, `weather`, `time_of_day` |

## Runtime (`sim_experiment/curriculum_runtime.py`)

`CurriculumController` is the state machine that consumes the definition:

- **Advancement is deterministic**: advance iff metric present in the
  evaluation aggregate AND `episodes_in_stage >= min_episodes` AND the
  threshold is met AND a next stage exists. A single lucky episode cannot
  advance a stage (min_episodes ≥ 1 enforced by validation).
- **Evaluation-driven**: decisions are made on the frozen-policy evaluation
  aggregate produced at `eval_frequency` — never on raw training reward.
- **Every decision is recorded** in `history` (stage, value, threshold,
  episodes, timestep, advanced flag) and written to `metrics.jsonl` under
  the `curriculum` scope plus `curriculum_state.json` in the run dir.
- **Metric aliasing**: curriculum names map to evaluation aggregates
  (`mean_return`→`mean_reward`, `lap_completion_rate`→`completion_rate`).
- **Stage scenario resolution**: `stage_scenario_dict()` resolves the
  stage's scenario from the standard library (or the manifest scenario)
  and applies `environment_overrides`.
- **Deterministic seeds**: `stage_seed(env_i) = base_seed + env_i + stage*10000`.

## Trainer integration (all algorithms)

On advancement the trainer builds fresh envs for the new stage and calls
`runner.set_envs(envs)` — the runner re-initializes rollout state at the
next update boundary (no torn rollouts). Env count is fixed across stages.

## Persistence & resume

- `curriculum_state.json` — stage index, episodes-in-stage, history,
  fingerprint; written on creation, every eval, and at run end.
- Checkpoints embed `curriculum_state` (trainer `extra` payload);
  checkpoint registry entries record `curriculum_stage_index`.
- **Resume**: restores controller state from the checkpoint. Fingerprint
  mismatch between checkpoint and manifest curriculum → `curriculum_mismatch`
  failure (safe, no silent restart). Missing state → fail unless
  `resume.restart_curriculum=true`.

## Validation

`validate_curriculum(dict, known_scenario_ids)` — structural checks:
stages exist, `scenario_id` known, `target_metric` resolves,
`min_episodes >= 1`, numeric threshold, unique stage_ids. Runs at
launch (non-retryable) and again inside the trainer.

## Constraints

- `env_mode="tcp"` + curriculum is rejected: headless TCP simulators
  cannot swap scenarios mid-run.
- Curriculum does not change the environment physics core — only the
  `ScenarioDefinition` bound to each env instance.
- Curriculum is PPO/SAC/DQN-compatible (shared harness), not a PPO
  special case.

## CLI

```
python -m sim_experiment.cli curriculum <experiment_id> <run_id>
```

Prints the live `curriculum_state.json`.
