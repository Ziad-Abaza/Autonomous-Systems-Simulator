---
noteId: "14a1a880bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Run Manager

**File:** `sim_experiment/run.py`

An **experiment** describes what should be executed; a **run** is one
actual execution. Runs are append-mostly records inside the experiment
directory and are never deleted by the platform.

## Run lifecycle

```
CREATED → QUEUED → STARTING → RUNNING → COMPLETED
                                  ↘ PAUSED ↗
                                  ↘ FAILED | CANCELLED | INTERRUPTED
```

- Transitions are **validated** (`_TRANSITIONS` table); invalid jumps
  raise `RuntimeError`.
- Terminal states (`COMPLETED`, `FAILED`, `CANCELLED`, `INTERRUPTED`)
  accept no further transitions.
- `INTERRUPTED` = trainer crash, timeout (`max_wall_seconds`), or
  infrastructure failure — distinct from a user `CANCELLED`.

## Run record (`run.json`)

```json
{
  "run_id": "run_20261003_105309_56e3b11f",
  "experiment_id": "exp_...",
  "status": "RUNNING",
  "seed": 42,
  "pid": 12345,
  "current_timestep": 40960,
  "episode_count": 37,
  "latest_metrics": {"sps": 331.5, "policy_loss": 0.02},
  "checkpoints": ["checkpoints/policy_40960.pt"],
  "evaluation_results": ["evaluation/eval_40000.json"],
  "replays": ["replays/eval_40000_ep0.json"],
  "resume_from": {"parent_run_id": "run_...", "checkpoint": "..."},
  "error": null,
  "exit_code": null,
  "trainer": "sim_experiment.trainers.ppo_trainer",
  "env_mode": "inprocess",
  "num_envs": 1
}
```

## API

```python
rmg = RunManager()
run = rmg.create_run(exp_dir, seed=42, experiment_id=..., trainer=...,
                     env_mode="inprocess", num_envs=1, resume_from=None)
rmg.set_status(exp_dir, run_id, RunStatus.QUEUED)
rmg.update_progress(exp_dir, run_id, timestep=1000, episode=5,
                    latest_metrics={"mean_reward": 12.3})
rmg.add_artifact_ref(exp_dir, run_id, "checkpoint", "checkpoints/p.pt")
rmg.list_runs(exp_dir)        # lightweight summaries
rmg.load_run(exp_dir, run_id) # full record
```

## Write discipline

- **Only the orchestrator writes `run.json`.** Trainers never touch it —
  they report through `metrics.jsonl`, `artifacts/registry.jsonl`, and
  `run_result.json`; `poll()` merges artifact refs into the run record.
- `run.json` writes are atomic (`tmp` + `os.replace`).
- Resuming a run creates a **new run** with `resume_from.parent_run_id` —
  history is preserved.
