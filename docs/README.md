# Documentation — Autonomous Systems Simulator

Welcome to the documentation for **Simulation Studio**, the modular 3D
simulation platform for autonomous-vehicle research and reinforcement
learning. Everything here reflects the current implementation (`simulator
4.0.0`, TCP protocol `2.1`).

If you are new, start with the [Quick Start](getting-started/quickstart.md).

---

## Getting started

| Page | What you'll learn |
|---|---|
| [Installation](getting-started/installation.md) | Dependencies, supported setups, verify install |
| [Quick Start](getting-started/quickstart.md) | Shortest path to a running simulation + first agent |
| [First Simulation](getting-started/first-simulation.md) | Guided tour: track → drive → record → replay |

## Using the Studio

| Page | What you'll learn |
|---|---|
| [Home & Track Library](user-guide/studio-home.md) | Tracks, templates, recents, datasets, settings |
| [Simulation](user-guide/simulation.md) | Driving, cameras, HUD, obs inspector, dynamics lab |
| [Environment Inspector](user-guide/environment-inspector.md) | Every inspector tab, readiness validation |
| [Keyboard & Mouse](user-guide/keyboard-shortcuts.md) | Complete shortcut reference |

## Authoring & configuration

| Page | What you'll learn |
|---|---|
| [Track Editor](track-editor/track-editor.md) | Splines, control points, boundaries, entities, spawn |
| [Project Schema (.sim.json)](configuration/project-schema.md) | Every serialized field, defaults, migration |
| [Settings Reference](configuration/settings-reference.md) | Studio settings and where each option lives |

## Simulation internals

| Page | What you'll learn |
|---|---|
| [Sensors](sensors/sensors.md) | Camera, LiDAR, IMU, vehicle-state — outputs, params, noise |
| [Vehicle Dynamics](vehicle-dynamics/vehicle-model.md) | Tire model, parameters, collision, validation lab |
| [Performance](performance/performance.md) | Measured throughput, sensor cost, tuning guidance |

## AI & reinforcement learning

| Page | What you'll learn |
|---|---|
| [RL Environment](reinforcement-learning/rl-overview.md) | Observation/action/reward/termination contract |
| [Training](reinforcement-learning/training.md) | PPO/SAC/DQN/BC trainers, env modes, curriculum |
| [External Agents](agents/external-agents.md) | Connect your own model — Python SDK & gymnasium |
| [TCP Protocol](agents/tcp-protocol.md) | NDJSON wire reference for any language |

## Research workflow

| Page | What you'll learn |
|---|---|
| [Experiments](experiments/experiments.md) | Manifests, runs, batches, workers, evaluation |
| [CLI Reference](experiments/cli-reference.md) | Every `sim_experiment.cli` subcommand + tools |
| [Datasets](datasets/datasets.md) | `transitions_v1` format, export, validation, splits |
| [Recording & Replay](recording-replay/recording-and-replay.md) | Episode capture and playback |

## Project

| Page | What you'll learn |
|---|---|
| [Architecture](architecture/architecture.md) | Package map, data flow, design principles |
| [Development](development/development.md) | Repo layout, tests, extension points |
| [Standalone Build](development/standalone-build.md) | PyInstaller Windows packaging |
| [Troubleshooting](troubleshooting/troubleshooting.md) | Symptom → cause → fix |
| [Glossary](reference/glossary.md) | Terminology reference |
| [FAQ](reference/faq.md) | Common questions |
| [Compatibility](reference/compatibility.md) | Versions, protocols, schema compatibility |

---

### Internal note

The `docs/` directory also contains flat `*.md` files written during earlier
development phases (audits, schemas, plans). Those are internal working
documents — kept for reference, untracked by git — while the folders above are
the maintained user-facing documentation.
