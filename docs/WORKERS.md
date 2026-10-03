---
noteId: "0747cc00bf0a11f1a29f1fbaabbd87c8"
tags: []

---

# Remote Worker Architecture

`sim_experiment/remote_worker.py` — **LAN-oriented**, token-authenticated,
versioned TCP workers. This is a control plane for dispatching runs, not
a distributed environment transport.

## Design

```
BatchScheduler ──> RemoteWorkerAdapter ──TCP/NDJSON──> WorkerService
                                                          └── LocalTrainingOrchestrator
                                                                   └── trainer subprocess
```

- `WorkerService(host, port, experiments_root, token)` wraps a
  `LocalTrainingOrchestrator`. Trainer subprocesses run **on the worker
  machine** against its own (or shared) `experiments_root`.
- `RemoteWorkerAdapter(host, port, token)` is a scheduler worker adapter —
  one short-lived connection per RPC.
- Wire format: single-line JSON request, single-line JSON response.

## Protocol

Every message: `{"v": "1.0", "type": ..., "token": ...}`

| type | payload | response |
|---|---|---|
| HELLO | — | HELLO_ACK {protocol_version, supported, capabilities} |
| LAUNCH | manifest(dict), experiment_dir, trainer, env_mode, run_overrides | LAUNCH_ACK {run_id} |
| POLL | experiment_dir, run_id | POLL_ACK {summary} |
| CANCEL | experiment_dir, run_id | CANCEL_ACK |

Errors: `auth_failed` (checked before anything else),
`protocol_version_mismatch` (supported: `1.0`), `launch_rejected`,
`unknown_type`.

Worker capabilities mirror the local shape (`type: "remote"`, trainer
list, `max_envs_per_run`) so scheduling decisions are transport-agnostic.

## Security scope

The token gate is a shared-secret LAN mechanism — it is **not** safe for
internet exposure. There is no TLS, no per-client identity, no rate
limiting. Deploy behind a trusted LAN or tunnel.

## Constraints

- The worker's `experiments_root` must contain the experiment (shared
  filesystem or pre-seeded replica); the manifest is sent inline in
  LAUNCH but environment/scenario JSON files are read from the worker's
  disk.
- `env_mode="tcp"` remains incompatible with curriculum runs regardless
  of worker type.
- No auto-discovery, no worker pools across machines beyond what the
  scheduler's worker list provides.

## CLI

```
# worker side
python -m sim_experiment.cli worker-serve --host 0.0.0.0 --port 9100 \
    --token <secret> --worker-root experiments

# scheduler side (probe)
python -m sim_experiment.cli worker-status --host <ip> --port 9100 --token <secret>
```

To schedule on remote workers, construct the scheduler explicitly:

```python
worker = Worker(worker_id="remote_1",
                capabilities=adapter.capabilities())
worker.adapter = RemoteWorkerAdapter(host, port, token)
s = BatchScheduler(root, workers=[worker, *local_workers])
```
