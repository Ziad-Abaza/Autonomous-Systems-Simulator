---
noteId: "42727aa0bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Batch Experiments

**File:** `sim_experiment/batch.py`

Deterministic parameter expansion — NOT a hyperparameter optimization
engine. Expansion is an inspectable cross product:

```python
expand_run_specs(manifest, seeds=[1,2,3])          # 3 runs, same scenario
expand_run_specs(manifest, seeds=[7],
                 scenario_ids=["basic_lane_following",
                               "high_speed_racing"])  # 2 runs
# cross product: seeds x scenario_ids, sorted deterministically
```

Each spec `{seed, scenario_id}` becomes an independent run with its own
run directory, metrics stream, and artifacts. Batch runs do **not**
mutate the experiment manifest — identity stays fixed; runs vary.

## CLI

```powershell
python -m sim_experiment.cli batch <experiment_id> --seeds 1 2 3
python -m sim_experiment.cli batch <experiment_id> --seeds 7 `
    --scenarios basic_lane_following high_speed_racing
python -m sim_experiment.cli launch <experiment_id> --trainer ppo   # per run
```

Batch-created runs start `CREATED`; launch each with the orchestrator
(or drive them programmatically).

## Comparing experiments

- `exp_mgr.list_experiments()` — side-by-side experiment summaries.
- `MetricsReader(run metrics).aggregate("episode")` — mean/min/max per
  metric for a run.
- `EvaluationResult.aggregate` — comparable eval results across runs.
- Identity fingerprints tell you *which* two experiments are actually
  comparable (same env fingerprint + seed structure, different training
  config) vs. apples-to-oranges.
