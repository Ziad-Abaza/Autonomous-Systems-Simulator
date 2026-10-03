---
noteId: "2d0c92e0bf0011f1a29f1fbaabbd87c8"
tags: []

---

# External Trainer Contract — v1.0

**File:** `sim_experiment/trainer_contract.py`
(`TRAINER_CONTRACT_VERSION = "1.0"`)

The contract is the machine-readable interface between the platform and
an external trainer process. It is materialized as
`runs/<run_id>/contract.json` before the trainer launches.

## Contract fields

```json
{
  "contract_version": "1.0",
  "experiment_id": "exp_...",
  "experiment_fingerprint": "...",
  "run_id": "run_...",
  "environment_name": "...",
  "environment_version": "1.0.0",
  "environment_fingerprint": "...",
  "scenario_id": "basic_lane_following",
  "scenario": { ...full scenario dict... },
  "seed": 42,
  "simulator_version": "4.0.0",
  "protocol_version": "2.0",
  "observation_schema": { ... },
  "action_schema": { ... },
  "training": { "algorithm": "ppo", "total_timesteps": ..., "algorithm_config": {...} },
  "evaluation": { "eval_seeds": [...], "num_episodes": 5, "deterministic_policy": true },
  "env_mode": "inprocess | tcp",
  "tcp": { "host": "127.0.0.1", "ports": [5555] },
  "paths": {
    "experiment_dir": "...", "run_dir": "...",
    "environment_json": "...", "scenario_json": "...",
    "metrics_file": "runs/.../metrics.jsonl",
    "checkpoints_dir": "...", "evaluation_dir": "...",
    "trajectories_dir": "...", "replays_dir": "...", "logs_dir": "...",
    "run_result": "runs/.../run_result.json"
  },
  "resume": { "parent_run_id": "...", "checkpoint": "..." } | null,
  "capabilities": ["metrics_jsonl", "checkpoints", "evaluation", "trajectories", "replays"]
}
```

## What a trainer must do

1. Parse `--run-dir`, load `contract.json`, call `validate_contract()`.
2. Build environment(s): `inprocess` → `build_env_from_dicts`;
   `tcp` → `SimGymEnv(host, port)` per port.
3. Write **metrics** to `paths.metrics_file` as scoped JSONL records
   (`step` / `episode` / `evaluation` / `run`). Never write `run.json`.
4. Write **checkpoints** into `paths.checkpoints_dir` and register each
   via `ArtifactRegistry(run_dir).register("checkpoint", rel_path, ...)`.
   Checkpoint payloads are opaque to the platform.
5. Optionally write evaluation results to `paths.evaluation_dir`,
   trajectories to `paths.trajectories_dir`, replays to `paths.replays_dir`.
6. At exit, write `paths.run_result`:

```json
{"status": "completed",
 "total_timesteps": 50000, "episodes_completed": 137,
 "wall_clock_time": 151.3, "sps": 330.6}
```

or on failure:

```json
{"status": "failed",
 "error": {"type": "trainer_exception", "message": "...", "traceback": "..."}}
```

## Compatibility

- `validate_contract()` rejects unknown `contract_version` values —
  forward-incompatible contracts fail loudly instead of silently.
- TCP protocol negotiation is a separate layer (`sim_net` handshake):
  clients declare `protocol_versions`; the server picks the highest
  mutual version or returns `protocol_version_mismatch`.
- The trainer must not import `sim_ui` or `sim_render`.

## Registering a new trainer

1. Create `sim_experiment/trainers/<name>_trainer.py` with a
   `main()` that implements the contract.
2. Add an entry to `TRAINER_MODULES` in
   `sim_experiment/orchestrator.py`.
3. No changes to the simulation core are permitted.
