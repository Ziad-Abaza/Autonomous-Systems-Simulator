---
noteId: "068e4000bf0a11f1a29f1fbaabbd87c8"
tags: []

---

# Batch Scheduler & Worker Pool

`sim_experiment/scheduler.py` adds a real execution layer on top of the
Phase 4 batch *expansion* (`sim_experiment/batch.py`, still used to build
run specs).

## Architecture

```
BatchScheduler
  └── workers[] (Worker: worker_id, capabilities, status, adapter)
        ├── LocalAdapter ──> LocalTrainingOrchestrator ──> trainer subprocess
        └── RemoteWorkerAdapter ──TCP/NDJSON──> WorkerService ──> orchestrator
```

- **Worker** = one execution slot, at most one run at a time. States:
  IDLE / STARTING / RUNNING / FAILED / STOPPING / OFFLINE.
- **Adapter** = the execution backend. `LocalAdapter` delegates to the
  in-process orchestrator; `RemoteWorkerAdapter` calls a TCP worker
  service (see `docs/WORKERS.md`).
- **Job** = one run spec (`{seed, scenario_id, algorithm_config}`) +
  attempt history. States: QUEUED / RUNNING / COMPLETED / FAILED /
  CANCELLED.

## Scheduling semantics

- `tick()` = one dispatch+poll round: fill idle workers from the queue,
  poll running jobs through their adapter, finalize transitions.
- `run_until_complete(batch_id, timeout_s)` drives ticks until all jobs
  are terminal, then writes `batch_result.json`.
- `cancel_batch` cancels running attempts through the adapter and marks
  queued jobs CANCELLED.

## Failure isolation & retries

One failed run never kills the batch. On a terminal FAILED/INTERRUPTED
attempt, `is_retryable_error` classifies the error:

- **Retryable** (fresh attempt queued, bounded by `RetryPolicy.max_retries`):
  `trainer_crash`, `trainer_exception`, `launch_failure`, `timeout`,
  `worker_crash`.
- **Never retried**: `invalid_contract`, `invalid_curriculum`,
  `unsupported_algorithm`, `curriculum_mismatch`, `incompatible_environment`,
  `protocol_version_mismatch`, `launch_rejected`, unknown types.

Every attempt is recorded (run_id, worker_id, status, error, timestamps) —
a retry is a NEW run, never silent continuation.

## Persistence

```
<experiment_dir>/batches/<batch_id>/batch.json        # live state, updated per transition
<experiment_dir>/batches/<batch_id>/batch_result.json # immutable final result
```

`batch_result.json` contains per-job: spec, status, run_ids, attempts,
retries, latest metrics, timestep/episode counts, artifact refs.

## Per-run overrides

`run_overrides` on `orchestrator.launch` (seed / scenario_dict /
algorithm_config) are applied to the *contract*, never the immutable
manifest. Scenario overrides resolve through the standard scenario
library.

## CLI

```
python -m sim_experiment.cli batch-run <exp_id> --seeds 1 2 3 \
    --workers 4 --trainer ppo --max-retries 1 --timeout 1800
```
