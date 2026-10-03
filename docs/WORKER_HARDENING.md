---
noteId: "b138a780bf1311f1a29f1fbaabbd87c8"
tags: []

---

# Worker Hardening

Phase 6 additions to the remote-worker execution layer. The security
model is unchanged: shared-token auth on every message, protocol-version
negotiation, LAN-oriented.

## Worker identity & registration

`WorkerService` instances carry a stable `worker_id`
(`worker_<uuid>`), advertised in `HELLO_ACK` capabilities and returned by
`REGISTER`. `RemoteWorkerAdapter.register()` / `handshake()` /
`heartbeat()` populate `adapter.worker_id`.

`WorkerRegistry` (`experiments_root/workers/registry.json`) persists
every worker that registers: capabilities, status, last heartbeat,
offline reason. Registration is idempotent — a worker reconnecting with
the same id reuses its record, and the registry survives scheduler
restarts.

## Heartbeats & leases

`BatchScheduler(heartbeat_interval_s=10, lease_ttl_s=30)` probes every
worker whose adapter exposes `heartbeat()` once per interval:

- success → `last_heartbeat` refreshed, missed counter reset, registry
  heartbeat recorded
- failure → `missed_heartbeats` increments
- silence past `lease_ttl_s` → worker → `OFFLINE`

An unreachable `poll` counts toward the same lease. `OFFLINE` workers are
excluded from dispatch.

## Job reclaim

When a worker goes OFFLINE, its in-flight job's current attempt is marked
`FAILED {type: "worker_crash"}` and the job returns to `QUEUED` — this
does **not** consume the retry budget (infrastructure loss ≠ attempt
failure). A healthy worker picks the job up on the next tick; the
attempts list keeps the full audit trail.

Duplicate-dispatch guard: a job whose latest attempt is still `RUNNING`
is never re-dispatched even if its job status regresses to `QUEUED`
(crash-consistent state) — the lease must die first.

## Retry classification

`is_retryable_error(error, retryable_types=...)` consults the batch's
`RetryPolicy.retryable_error_types` (previously ignored — only the module
set applied). `NON_RETRYABLE_ERROR_TYPES` always wins: a policy cannot
make config/protocol errors retryable.

## Path confinement (security)

`WorkerService` root-checks `experiment_dir` on **LAUNCH, POLL, and
CANCEL**: the directory must resolve under the worker's
`experiments_root`. Escapes (including cross-drive on Windows) return
`invalid_experiment_dir` before any filesystem work.

## STATUS message

`STATUS` returns worker liveness, `worker_id`, protocol version, and
capabilities — for operator inspection without side effects.

## CLI

```powershell
python -m sim_experiment.cli worker-serve --token $T --worker-root experiments
python -m sim_experiment.cli worker-register --port 9100 --token $T
python -m sim_experiment.cli worker-list --port 9100 --token $T
python -m sim_experiment.cli batch-run <exp> --worker 127.0.0.1:9100 --token $T
```
