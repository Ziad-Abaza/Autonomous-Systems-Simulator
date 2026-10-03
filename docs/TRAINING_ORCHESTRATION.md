---
noteId: "2cb17a90bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Training Orchestration

**File:** `sim_experiment/orchestrator.py` — `LocalTrainingOrchestrator`

Launches, monitors, and terminates external trainer processes for runs.

## Launch flow

```
orch.launch(manifest, experiment_dir, trainer="ppo", env_mode="inprocess")
  1. Resolve trainer -> module under sim_experiment.trainers.* (whitelist)
  2. RunManager.create_run() -> run_<ts>_<hash>/ CREATED
  3. env_mode == "tcp": HeadlessSimProcessPool.start() -> allocated ports
  4. build_contract() -> validate_contract() -> contract.json
  5. CREATED -> QUEUED -> STARTING
  6. Popen([python, "-m", "sim_experiment.trainers.<t>", "--run-dir", rd])
       stdout -> logs/stdout.log   stderr -> logs/stderr.log
  7. STARTING -> RUNNING (pid recorded)
```

## Monitoring (`poll()`)

Per call:

1. Tail `metrics.jsonl` → `current_timestep`, `episode_count`,
   `latest_metrics` merged into `run.json`.
2. Sync `artifacts/registry.jsonl` → `run.checkpoints`,
   `run.evaluation_results`, `run.replays`.
3. `max_wall_seconds` exceeded → terminate → `INTERRUPTED`
   (`error.type = "timeout"`).
4. Process exited → `_finalize_run()`:
   - `run_result.json.status == "completed"` → `COMPLETED`
   - `run_result.json.status` in {failed, cancelled, interrupted} → same
   - no result file + exit 0 → `COMPLETED`
   - no result file + nonzero exit → `FAILED`
     (`error.type = "trainer_crash"`)

`wait(exp_dir, run_id, timeout_s)` polls until terminal.

## Cancellation

`cancel()` → `proc.terminate()` → 5 s grace → `proc.kill()` →
`CANCELLED`. Partial artifacts (metrics, checkpoints, logs) remain on
disk; the run record is preserved.

## Resume

`resume` never mutates an old run: a **new run** is created with

```python
resume_from = {"parent_run_id": old_run_id,
               "checkpoint": "<abs path or run-relative>"}
```

The trainer resolves the checkpoint (run-relative, then
`runs/<parent_run_id>/`-relative) and continues timestep counting via
`PPORunner.global_step_offset`.

## env_mode

| Mode | How envs are created |
|------|----------------------|
| `inprocess` | Trainer builds N `SimulationEnvironment` instances from `environment.json` + `scenario.json` (seeds `seed+i`) |
| `tcp` | Orchestrator spawns `HeadlessSimProcessPool` (N × `main.py --headless --port`); contract carries `tcp.host`/`tcp.ports`; trainer connects via `SimGymEnv` |

## Safety model

- Trainer modules restricted to `sim_experiment.trainers.*` — no
  arbitrary command execution; subprocess invoked as an arg list, never
  a shell string.
- Run directories must be inside the experiments root
  (`_check_run_dir`).
- Contract must validate before a process is spawned.
- Orchestrator is the sole writer of `run.json`.
