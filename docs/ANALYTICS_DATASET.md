---
noteId: "115c0ee0bf0a11f1a29f1fbaabbd87c8"
tags: []

---

# Experiment Analytics & Dataset Export

## Analytics (`sim_experiment/analytics.py`)

Read-only analytics over the existing `metrics.jsonl` artifact — no new
metrics system, experiment/run data stays immutable.

- `load_metrics_series(run_dir, scope, metric)` → sorted `[(timestep, value)]`
- `smooth(series, window)` → trailing moving average (chart-ready)
- `compare_runs(run_dirs, metric, scope, smooth_window)` →
  `{metric, scope, series: [{label, raw, smoothed, max, min, final, mean, count}]}`
- `compare_experiments(experiment_dirs, ...)` → same, labeled `exp/run`
- `metric_summary(run_dir)` → one-line summary (episodes, mean/best/final
  return, eval peaks, timesteps, sps, termination-reason histogram)
- `list_run_dirs(experiment_dir)`

CLI:

```
python -m sim_experiment.cli compare <exp_id> --metric reward \
    --scope episode --smooth 10 [--runs run_a,run_b]
```

## Trajectory explorer (`sim_experiment/dataset.py::list_episodes`)

Episode-level index over `trajectories/*.jsonl`: episode_id, fingerprint,
scenario, seed, steps, total_return, termination_reason.

```
python -m sim_experiment.cli trajectories <exp_id> <run_id> [--json]
```

## Dataset export (`sim_experiment/dataset.py::export_dataset`)

Writes `transitions_v1`:

```
<dest>/manifest.json    — format, schema_hash, env_fingerprint, filters, skipped counts
<dest>/episodes.jsonl   — {episode_id, seed, env_fingerprint, scenario_id,
                           total_return, length, termination_reason,
                           steps: [{obs, action, reward, terminated, truncated,
                                    termination_reason}]}
```

### Isolation guarantees

Steps contain **only** `agent_data` fields (whitelisted). `diagnostic_data`
(speed, lateral_offset, collision flags, ...) is structurally excluded —
it can never reach the export.

### Schema-compat enforcement

- `env_fingerprint` filter: episodes from other environments are skipped
  and counted (`skipped.fingerprint_or_schema_mismatch`).
- Observation/action schema hash is computed per episode; mismatches are
  skipped — never silently mixed.
- `min_return` / `termination_reasons` filters apply at episode level.

CLI:

```
python -m sim_experiment.cli dataset-export <exp_id> <run_id> --dest out \
    [--min-return 100] [--env-fingerprint <fp>] [--termination-reasons a,b]
```

This format is the baseline for imitation-learning ingestion
(obs→action transition tuples); BC/GAIL wrappers can consume it directly.
