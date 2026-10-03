# Development Guide

Repo layout, development setup, test/QA harnesses, and the code-level extension points for Simulation Studio.

## Repository layout

```text
Simulation/
├── main.py                          # launcher: interactive studio or headless TCP server
├── sim_version.py                   # SIMULATOR_VERSION = "4.0.0"
├── AI_Environment_Simulator.spec    # PyInstaller spec (mirrors build/package_windows.py)
├── README.md
├── sim_core/                        # physics & geometry foundation (no RL concepts)
│   ├── vehicle/                     #   vehicle_model.py, vehicle_config.py, collision.py
│   ├── track/                       #   road_definition.py, spline.py, mesh_generator.py, track_queries.py
│   ├── sensors/                     #   base_sensor.py + vehicle_state/raycast/imu/camera + sensor_manager.py
│   ├── collision/                   #   spatial_hash.py (OBB2D SAT lives in vehicle/collision.py + track code)
│   ├── world/                       #   entity.py (WorldEntity + ENTITY_CLASS_MAP), obstacle.py, checkpoint.py
│   └── clock.py, math_utils.py      #   FixedClock (dt=1/60, seeded RNG), Vec2/Vec3
├── sim_env/                         # environment lifecycle + dual config paths (legacy blocks vs AgentDefinition)
│   ├── environment.py               #   SimulationEnvironment reset()/step()
│   ├── spaces.py, reward_engine.py, termination_engine.py,
│   │   domain_randomizer.py, scenarios.py, episode_config.py     # legacy path
│   ├── agent.py, observation_designer.py, action_designer.py,
│   │   reward_designer.py, termination_designer.py,
│   │   randomization_designer.py, scenario_designer.py,
│   │   sensor_config.py             # declarative path
│   ├── observation_contract.py, curriculum.py, validator.py,
│   │   versioning.py, templates.py, experiment.py, export.py
├── sim_render/                      # ModernGL renderer: renderer.py, camera.py (CHASE/HOOD/TOP_DOWN/ORBIT),
│                                    #   mesh.py, offscreen.py (FBO for camera_rgb), shaders.py
├── sim_ui/                          # pygame + ModernGL studio (single process, hosts a TCP server)
│   ├── app.py                       #   SimulationStudioApp — screen/tab state, server poll, recording, training services
│   ├── screens/                     #   home_screen, workspace_screen, datasets_panel, dynamics_panel,
│   │                                #   experiments_panel, dialogs
│   ├── editor.py, editor_ui.py, edit_history.py   # track editor + undo (64-deep snapshots)
│   ├── inspector.py                 #   two-tier inspector: GEO + RL tabs
│   ├── hud.py, theme.py, widgets.py, ui_overlay.py, thumbnails.py, train_providers.py
├── sim_net/                         # NDJSON TCP protocol 2.1: protocol.py, server.py (1 env),
│                                    #   multi_server.py (1 port / N envs), env_handler.py (session)
├── sim_client/                      # client.py (SimulationClient), gym_env.py (SimGymEnv),
│   └── agents/                      #   pid_driver, random_agent, ppo_train (demo),
│                                    #   ppo_baseline, sac_baseline, dqn_baseline, replay_buffer
├── sim_recorder/                    # recorder.py (telemetry episode recording), replay.py (player)
├── sim_project/                     # serializer.py (.sim.json schema 2.0.0/3.0.0), library.py
│   │                                #   (tracks/ + presets/ roots), settings.py, presets/ (factories)
├── sim_experiment/                  # experiment platform
│   ├── manifest.py, manager.py, run.py, metrics.py, artifacts.py
│   ├── trainer_contract.py, capabilities.py, orchestrator.py, headless.py
│   ├── vec_env.py, process_env.py, batch.py, scheduler.py
│   ├── remote_worker.py, worker_registry.py, evaluation.py
│   ├── dataset.py, dataset_inspect.py, trajectory.py, trajectory_sampler.py
│   ├── curriculum_runtime.py, convergence.py, analytics.py, reproduce.py
│   ├── bc/                          #   BCRunner, BCPolicy, dataset loading
│   ├── trainers/                    #   ppo/sac/dqn/bc/dummy trainers + _harness.py
│   └── cli.py                       #   python -m sim_experiment.cli
├── agentRL/                         # experimental continual-RL library (v0.1.0, partial)
├── tools/                           # ui_shots.py, smoke_interactions.py, tcp_recording_e2e.py,
│   └── vehicle_dynamics/            #   calibrate.py, harness.py, maneuvers.py, run_validation.py
├── tests/                           # 78 files / 542 tests (incl. tests/agent/)
├── benchmarks/                      # committed results (phase4, vision, broadphase) + runners
├── build/                           # package_windows.py (PyInstaller driver)
├── docs/                            # curated docs tree (this file) + flat technical references
├── experiments/                     # default experiments root: manifests + runs/ + batches/ + workers/
├── presets/                         # bundled read-only .sim.json library root
├── tracks/                          # user track library (gitignored)
├── data/                            # recordings/, datasets/, studio_settings.json (gitignored)
├── logs/                            # trainer stdout/stderr, headless sim_<port>.log
├── assets/screenshots/qa/           # ui_shots.py output (gitignored)
├── dist/, build_temp/               # PyInstaller outputs (gitignored)
└── last_episode.json                # legacy episode recording, REPLAY-tab fallback (gitignored)
```

## Setup

- **Python**: 3.13 (developed on 3.13.7).
- **There is no `requirements.txt` / `pyproject.toml`.** Install dependencies manually:

```powershell
pip install numpy pygame moderngl opencv-python gymnasium
pip install torch        # only needed for trainers, RL baselines, agentRL
pip install pytest       # only needed for tests
pip install pyinstaller  # only needed for the standalone build
```

Notes on what is actually imported:

| Dependency | Where used |
|---|---|
| `numpy`, `pygame`, `moderngl` | everywhere (core, UI, renderer) |
| `opencv-python` (`cv2`) | `sim_core/sensors/camera_sensor.py` only |
| `gymnasium` | `sim_client/gym_env.py` (`import gym` fallback exists) |
| `torch` | `sim_client/agents/*_baseline.py`, `sim_experiment` trainers + `bc/`, `agentRL/algos`, `experiments/*.py` scripts |
| `pytest` | `tests/` |

The dev environment runs torch 2.10 (CPU), gymnasium 1.2.1, moderngl 5.12, pygame 2.6.1.

Run the app:

```powershell
python main.py                                          # interactive studio
python main.py --headless --port 8765                   # headless single-env TCP server
python main.py --headless --num-envs 4 --port 8765      # headless multi-env, one port
python main.py --track oval|serpentine|obstacle|<path.sim.json> --width 1280 --height 720
```

## Tests and QA harnesses

```powershell
python -m pytest tests/ -q                  # 542 tests / 78 files, incl. tests/agent/
python tools/smoke_interactions.py          # ~37 interaction checks through real app event handling
python tools/tcp_recording_e2e.py           # 15 TCP-driven recording E2E checks
python tools/ui_shots.py [W H]              # ~35 screenshots → assets/screenshots/qa/<W>x<H>_*.png
```

- `tests/agent/conftest.py` needs `tracks/*.sim.json`, which is gitignored — generate tracks by saving from the studio first, or those tests are skipped/fail locally.
- There are no pytest markers; the suite is one flat run.
- Benchmark scripts live under `benchmarks/` (`phase5_benchmarks.py`, `phase6/benchmark_runner.py`, `phase6/perf_runner.py`); committed results are `phase4_results.json`, `vision_results.json`, `broadphase_results.json`.

## Extension points

### New sensor

1. Subclass `BaseSensor` (`sim_core/sensors/base_sensor.py`): implement `sample(sim_time, context, rng)` returning an output dict; base class handles `update_frequency_hz` gating, latency buffer, and `noise_std`.
2. Register it in `sim_env/sensor_config.py`: add an entry to `_SENSOR_DEFAULTS` (params defaults), `_SENSOR_TYPE_NAMES` (display name), optionally `_NAME_TYPE_HINTS` (name→type inference), and the dispatch in `SensorConfig.build_sensor()`.
3. It then serializes under `agent.sensor_configs` and shows up in the inspector SENSORS tab.

### New observation channel

Add an `ObservationChannelConfig` to the agent's `ObservationSpaceDefinition` (`sim_env/observation_designer.py`): set `source_sensor` + `source_key` to keys the sensor emits, choose `channel_type`/`shape`/`normalization`/`range_low`/`range_high`. Flat vector order = channel declaration order. Keep `category="agent_observation"` — `debug_telemetry`/`oracle_ground_truth` channels trip the leakage guard (`validate_no_leakage` → validator ERROR, compiled pipeline raises `ValueError`).

### New reward component

Add a `component_type` branch in `CompiledRewardEngine.compute_step_reward` (`sim_env/reward_designer.py`, the `if ctype == ...` dispatch) and use it via `RewardComponentConfig`. Contribution = `raw × weight`; add to `create_default_racing_reward` only if it should ship in defaults. Params go through `comp.params.get(...)` — note existing params like `max_step_delta_m`/`tolerance` are declared but not consumed, so document any new params honestly.

### New termination condition

Add a constant to `TerminationConditionType` and a branch in `CompiledTerminationEvaluator.evaluate` (`sim_env/termination_designer.py`), then use it via `TerminationRuleConfig` (`is_truncation=True` for horizon limits). Precedent: `CUSTOM_THRESHOLD` is declared but **not** evaluated — the enum alone does nothing until `evaluate` handles it.

### New entity type

1. Subclass `WorldEntity` (`sim_core/world/entity.py`): set `entity_type`, `semantic_label`, `is_collidable`; override `get_obb`/`update`/`to_dict`/`from_dict` as needed.
2. Register an alias in `ENTITY_CLASS_MAP` (`entity.py`) so `entity_from_dict` can deserialize it.
3. Editor support: add the tool name to the `active_tool` placement dispatch in `sim_ui/editor.py` (`~line 234`) and a gizmo branch in `draw_editor` (`~line 415`); add a `+Type` place action in the inspector SCENE tab.

### New trainer

1. Create `sim_experiment/trainers/<name>_trainer.py` — launched as `python -m sim_experiment.trainers.<name>_trainer --run-dir <dir>` (the module name **must** start with `sim_experiment.trainers`; the orchestrator rejects others).
2. Register it in `TRAINER_MODULES` (`sim_experiment/orchestrator.py`) and `TRAINER_CAPABILITIES` (`sim_experiment/capabilities.py`: supported algorithms, action type, obs kinds, multi_env, eval support).
3. Honor `contract.json` (trainer contract `1.0`: env build honoring `env_mode`, paths, resume), write `metrics.jsonl` records `{scope, seq, ts, timestep, metrics}`, register artifacts, and always write `run_result.json` (`{"status": "completed"|"failed"|...}`) — a nonzero exit without it is finalized as `FAILED: trainer_crash`.
4. Use `trainers/_harness.py` for contract loading, env construction, action bounds, periodic eval, and trajectory recording.

### New scenario

Add an entry to `ScenarioDefinition.get_standard_scenarios()` (`sim_env/scenario_designer.py`). It becomes usable by `SET_SCENARIO {"scenario_id": ...}` over TCP (protocol ≥ 2.1), by manifests/`scenario_id`, and — after adding it to the inspector SCENARIO tab preset enum — in the UI.

### New template

Add a builder + `list_templates` entry in `sim_env/templates.py` (`EnvironmentTemplateManager.create_project_from_template`). It appears in the home TEMPLATES nav and the new-track dialog.

### New UI surface

Follow `sim_ui/screens/` patterns: screens are plain classes drawn through the custom widget layer — `UIContext` (`sim_ui/widgets.py`) owns hit regions via `ctx.hit(rect, action, ...)`; there is no imgui. Wire a new tab into `ws_tab` handling in `workspace_screen.py` and the inspector tiers in `inspector.py`. Palettes come from `theme.py` (`PALETTES`: dark, dark_ocean, dark_ember, light, light_solar).

## Conventions

- **Serialization**: dataclasses with `to_dict`/`from_dict`, tolerant `.get`-based loading — loading never gates on strictness; the validator reports issues (`IssueSeverity` ERROR/WARNING/INFO) instead of rejecting files.
- **Versioning**: SHA-256 fingerprints over `json.dumps(sort_keys)` with volatile keys excluded (`version`, `timestamp`, `last_saved`, `author`, `fingerprint`, …); `environment_version` auto-increments its patch component on fingerprint change at save.
- **Validators report, not gate**: `EnvironmentValidator.validate(...)` returns a `ValidationReport` (`is_valid_for_rl` false if any ERROR); the inspector re-runs it after every property change and badges the owning tab.
- **Tests colocated by feature** under `tests/` (mirroring package structure; `tests/agent/` covers agentRL).
- **Serialized vs runtime names differ** in places — e.g. the raycast sensor serializes as `lidar_rays` but reports `sensor_type="raycast_lidar"` at runtime; serialized sensor `local_pos` defaults differ from runtime constructor defaults. Check `sensor_config.py` before trusting either.

## In-repo flat docs

`docs/*.md` holds ~70 older technical references kept for context — e.g. `ACTION_SCHEMA.md`, `OBSERVATION_SCHEMA.md`, `SENSOR_CONFIGURATION.md`, `REWARD_CONFIGURATION.md`, `TERMINATION_SYSTEM.md`, `SCENARIO_SYSTEM.md`, `TRAINER_CONTRACT.md`, `TIME_MODEL.md`, `CLI_REFERENCE.md`, `TRAINERS.md`, `PARALLEL_ENVIRONMENTS.md` — plus phase reports/audits. They are working documents; where they conflict with the curated tree, trust the curated tree and the code.

## agentRL status

`agentRL/` is an experimental continual multi-track RL library (`v0.1.0`, partially implemented). Stable surface: `core/` (versions, `seed_tree`, configs), `obs/` (channel spec + `ObsEncoder`), `act/` (`ActionAdapter`), `rewards/` (presets `drive_v1`/`term_v1`), `envs/` (`TrackRegistry`, track generators, `EnvFactory`, `ScenarioMutator`), `memory/` (replay + track-rehearsal buffers). `algos/` (PPO/SAC agents), `train/` (trainer + continual orchestration), `eval/`, `checkpoints/`, `experiments/` are under active development; there is no packaged training CLI. See `agentRL/AGENT_RL_ARCHITECTURE.md` for the design doc.

## See also

- [Architecture](../architecture/architecture.md) — package map, data flow, process model
- [Standalone build](standalone-build.md) — PyInstaller packaging
- [Installation](../getting-started/installation.md)
- [Experiment CLI reference](../experiments/cli-reference.md)
- [Troubleshooting](../troubleshooting/troubleshooting.md)
- [Performance](../performance/performance.md)
