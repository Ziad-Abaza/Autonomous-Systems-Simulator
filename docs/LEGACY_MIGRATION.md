---
noteId: "d14d3d60bf1311f1a29f1fbaabbd87c8"
tags: []

---

# Legacy Migration Map

Deprecation policy for pre-contract training paths, updated in Phase 6.
Nothing is deleted — every legacy surface remains runnable — but only the
contract path is supported for new work.

## Deprecation map

| Legacy path | Status | Replacement | Notes |
|---|---|---|---|
| `sim_client/train.py` demo driver | **Deprecated** | `python -m sim_experiment.cli launch` | Skips manifest, fingerprints, metrics schema, run records. Kept for interactive demos; do not extend. |
| `PPORunner`/`SACRunner`/`DQNRunner` direct use | Supported as **runners** | `sim_experiment.trainers.*_trainer` | Runners are now VectorEnv-native (accept env lists, `SyncVectorEnv`, `ProcessVectorEnv`); external trainers wrap them for contract runs. |
| `sim_client/gym_env.py` direct scripts | Supported transport | contract `env_mode="tcp"` | The transport is canonical; contract wrapping adds isolation + lifecycle. |
| `sim_net/server.py` `SimulationServer` | Retained | `SimServerMulti` for multi-env | Single-client server remains the per-env headless backend. |
| Legacy env branch (`env.agent is None`) | Retained | `AgentDefinition` environments | Legacy projects/tests depend; new envs should use AgentDefinition. |
| `trajectory_episodes` int | Back-compat | `trajectory_sampling` dict | Int maps to `{"mode": "first_n", n}`. |

## Migration guide

- **Demo scripts → contract runs.** Replace `python -m sim_client.train`
  with `cli create` + `cli launch` (or `orch.launch`) to get fingerprints,
  metrics, checkpoints, and run records.
- **Env lists → VectorEnv.** Code that loops `for env in envs` can wrap the
  list in `SyncVectorEnv` to get `reset_all`/`step_all`/`set_scenario`
  semantics matching `ProcessVectorEnv` and TCP backends.
- **In-process → process isolation.** `env_mode="process"` gives the same
  determinism contract with real process boundaries; `env_mode="tcp"`
  crosses machines; `env_mode="tcp_multi"` hosts N envs in one sim process.
- **Run trajectories → datasets.** `dataset-export` + `dataset-validate`
  + `dataset-split` feed `train-bc`.

## Still supported (not deprecated)

- `SimulationServer` (single-client) — the per-env TCP backend.
- `HeadlessSimProcessPool` — now also `shared_process` mode.
- Legacy `ObservationSchema` — wire/compat layer, embedded in project
  files and the TCP handshake.
