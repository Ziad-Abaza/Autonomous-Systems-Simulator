---
noteId: "2d5534a0bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Metrics Schema

**File:** `sim_experiment/metrics.py`

Structured, append-only, buffered metrics storage for runs.
`metrics.jsonl` in each run directory — one JSON object per line.

## Record format

```json
{
  "scope": "step | episode | evaluation | run",
  "seq": 412,
  "ts": 1759484000.1234,
  "timestep": 40960,
  "metrics": { "...": "..." }
}
```

| Field | Meaning |
|-------|---------|
| `scope` | `step` (per-step debug), `episode` (episode-end stats), `evaluation` (eval aggregates), `run` (update/trainer-level) |
| `seq` | Monotonic sequence — continues across resume within the same file |
| `ts` | Wall-clock timestamp (metadata only) |
| `timestep` | Step index — the canonical ordering field |
| `metrics` | Arbitrary key→value map; NaN/Inf sanitized to `null` |

## Write discipline

- `MetricsWriter(path, buffer_size=64)` buffers records; disk I/O happens
  per-buffer or on `flush()`/`close()` — **never per simulation step**.
- Sequence numbering resumes from the file tail, so appending after a
  restart does not reset indices.
- `MetricsReader` skips a torn final line (crash-safe tail).

## Reader API

```python
r = MetricsReader("runs/.../metrics.jsonl")
r.read_all()            # all records
r.by_scope("episode")   # episode records only
r.latest()              # newest record
r.latest_metrics()      # newest metrics dict
r.tail(50)              # last N records
r.aggregate("episode")  # {key/mean, key/min, key/max, key/last}
```

## What the PPO trainer writes

| Scope | Keys |
|-------|------|
| `episode` | `reward`, `length`, `termination_reason`, `mean_lateral_error`, `mean_speed`, `checkpoints_passed` |
| `run` | `sps`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `explained_variance`, `episodes_completed` (plus `final` summary row) |
| `evaluation` | all `EvaluationResult.aggregate` keys |
| `step` | only when a trajectory/debug recorder requests it |

## Failure semantics

Non-finite metrics become `null` on write — a poisoned metric cannot
corrupt the JSONL stream or silently inject NaN into aggregations.
