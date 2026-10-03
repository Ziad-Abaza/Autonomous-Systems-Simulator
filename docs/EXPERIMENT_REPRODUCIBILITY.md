---
noteId: "432e2980bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Experiment Reproducibility

**File:** `sim_experiment/reproduce.py` — `check_reproducibility(exp_dir)`

Honest verification: the checker verifies every artifact required to
reproduce an experiment and reports exactly what can and cannot be
confirmed.

## Checks

| Check | Fatal |
|-------|-------|
| `experiment.json` exists and parses | yes |
| `manifest_version` supported | no |
| `environment.json` exists | yes |
| environment fingerprint recomputed == recorded | yes |
| `scenario.json` exists and == `scenario_configuration` | no |
| observation schema present | no |
| action schema present | no |
| termination config present | no |
| episode config present | no |
| `simulator_version` known to this build | no |
| `protocol_version` in `SUPPORTED_PROTOCOL_VERSIONS` | no |
| training config valid | no |

Report shape:

```json
{"reproducible": true,
 "exact_reproduction_guaranteed": false,
 "checks": [{"name": ..., "ok": true, "detail": "...", "fatal": false}],
 "fatal_failures": [],
 "warnings": []}
```

`exact_reproduction_guaranteed` is intentionally **always false**:
bit-exact reproduction additionally depends on torch nondeterminism,
BLAS threading, and hardware — the platform guarantees *configuration-
level* reproducibility (same env snapshot, same seed, same config), not
bitwise equality across machines.

## What travels with an experiment

- Full environment snapshot (`environment.json`) — not a reference
- Scenario snapshot (`scenario.json`)
- All schemas + episode/curriculum/randomization configs in the manifest
- Seed, simulator version, protocol version
- Runs: metrics, checkpoints, evaluations, replays, trajectories, logs

`cli export <exp> --dest <dir>` copies the whole directory → a
self-describing, self-verifying artifact bundle (`cli reproduce` works
on the copy).
