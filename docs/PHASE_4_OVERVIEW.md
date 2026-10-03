---
noteId: "13e7f570bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Phase 4 — Training & Experiment Platform Overview

Phase 4 turns the simulator from "an environment you can connect to" into
"a platform that can manage reproducible training workflows at scale."

## Goals

- Make experiments first-class entities (not ad-hoc scripts).
- Keep the training algorithm/trainer **outside** the simulation core.
- Introduce a Run Manager, orchestrator, metrics pipeline, evaluation
  system, artifact management, trajectories, batch expansion, headless and
  parallel environment support, UI dashboard, and CLI — all layered on
  top of the existing Phase 3 architecture without rewriting it.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │              UI / CLI                │
                    │   inspector TRAIN tab │ cli.py       │
                    └──────────┬──────────────────────────┘
                               │ service calls only
              ┌────────────────┼───────────────────────────┐
              │  ExperimentManager │ RunManager │ Orchestrator  │  (sim_experiment/)
              └────────────────┼───────────────────────────┘
                               │ contract.json + subprocess
              ┌────────────────▼───────────────┐
              │  External trainer process      │  python -m sim_experiment.trainers.*
              │  (PPO baseline, future SAC/DQN)│  — never inside sim_core/
              └────────┬───────────────────────┘
                       │ Gym-style API or TCP 2.0 (negotiated)
              ┌────────▼───────────────────────┐
              │ SimulationEnvironment(s)        │  sim_env + sim_core (unchanged core)
              └────────────────────────────────┘
```

## Key invariants

- `sim_core/` and `sim_env/` contain **no** training loop or RL algorithm.
- Experiment manifests are **immutable after first launch**; edits create
  a new identity.
- One `run.json` writer: the orchestrator. Trainers report through
  contract artifacts only.
- Determinism: seed travels from manifest → contract → env/trainer;
  environment and experiment fingerprints are SHA-256 over canonical
  JSON.
- Agent-facing data vs diagnostic data separation is enforced in the
  trajectory format, not just by convention.

## Directory layout

```
experiments/
└── exp_<16hex>/                # experiment (immutable snapshot)
    ├── experiment.json         # ExperimentManifest
    ├── environment.json        # full environment snapshot
    ├── scenario.json           # scenario snapshot
    └── runs/
        └── run_<ts>_<8hex>/
            ├── run.json        # lifecycle record
            ├── contract.json   # external trainer contract v1.0
            ├── metrics.jsonl   # structured metrics stream
            ├── run_result.json # trainer exit report
            ├── logs/           # stdout.log / stderr.log
            ├── checkpoints/    # model artifacts (+ artifacts/registry.jsonl)
            ├── evaluation/     # eval_<step>.json results
            ├── replays/        # replay artifacts
            └── trajectories/   # per-episode JSONL datasets
```

## Components

| Module | Responsibility |
|--------|----------------|
| `sim_experiment/manifest.py` | `ExperimentManifest`, `TrainingConfig`, `EvaluationConfig`, identity fingerprints |
| `sim_experiment/manager.py` | `ExperimentManager` — create/load/list/clone/launch/archive/export |
| `sim_experiment/run.py` | `Run`, `RunStatus`, validated transitions, `RunManager` |
| `sim_experiment/orchestrator.py` | `LocalTrainingOrchestrator` — launch/poll/cancel/wait/resume |
| `sim_experiment/trainer_contract.py` | Contract v1.0 build + validation |
| `sim_experiment/metrics.py` | `MetricsWriter`/`MetricsReader`, scoped buffered JSONL |
| `sim_experiment/artifacts.py` | `ArtifactRegistry` — checkpoint/eval/replay refs |
| `sim_experiment/evaluation.py` | `evaluate_policy`, `EvaluationResult`, PPO checkpoint adapter |
| `sim_experiment/trajectory.py` | `TrajectoryWriter`/`Reader` — agent vs diagnostic separation |
| `sim_experiment/batch.py` | deterministic seeds×scenarios expansion |
| `sim_experiment/reproduce.py` | `check_reproducibility` — honest verification |
| `sim_experiment/headless.py` | env factory, `HeadlessEnvPool`, `HeadlessSimProcessPool` |
| `sim_experiment/cli.py` | `python -m sim_experiment.cli` — full workflow CLI |
| `sim_experiment/trainers/` | external trainer entry points (`ppo_trainer`, `dummy_trainer`) |

## Quick start

```powershell
python -m sim_experiment.cli validate-env --project presets/oval_circuit.sim.json
python -m sim_experiment.cli create --project presets/oval_circuit.sim.json `
    --name baseline --timesteps 50000
python -m sim_experiment.cli launch <experiment_id> --trainer ppo
python -m sim_experiment.cli status <experiment_id> <run_id> --watch
python -m sim_experiment.cli evaluate <experiment_id> <run_id>
python -m sim_experiment.cli reproduce <experiment_id>
```
