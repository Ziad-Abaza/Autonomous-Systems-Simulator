---
noteId: "de589470bef811f1a29f1fbaabbd87c8"
tags: []

---

# Phase 4 Audit Report: Training & Experiment Platform Baseline

**Date:** October 2026
**Auditor:** Full-codebase review (every production module inspected, all tests executed)
**Test Result:** 90/90 PASSED in ~15s (baseline grew from 87 → 90 since the Phase 3 audit)
**Git Head:** `aa54d68` (unit/integration tests for designers)

---

## 1. Current Architecture

The platform is a pure-Python project (stdlib + numpy + pygame + moderngl + torch for agents). No build system, no database, no external services.

```
sim_core/      physics, track spline, sensors, world entities, collision (SpatialHashGrid2D)
sim_env/       SimulationEnvironment + Phase 3 declarative designers + versioning/validation/export
sim_render/    ModernGL 3D renderer + offscreen FBO (camera sensor)
sim_ui/        pygame app: modes SIM / EDITOR / REPLAY + EnvironmentInspector
sim_net/       TCP NDJSON server (protocol 2.0) + client SDK
sim_client/    SimulationClient, SimGymEnv (gymnasium.Env), agents (pid, random, ppo_train, ppo_baseline)
sim_recorder/  EpisodeRecorder (JSON/gz frames) + EpisodeReplayPlayer
sim_project/   EnvironmentProject serializer (*.sim.json, schema 1.0→3.0 migration) + presets
experiments/   two completed PPO baselines + runner scripts (ad-hoc artifacts)
benchmarks/    broadphase + vision benchmarks (perf scripts, not wired to training)
tests/         26 test files, 90 tests
```

### Runtime pipeline (verified, `sim_env/environment.py` — 585 lines)

- Dual-path: Phase 3 compiled pipelines (`CompiledActionDecoder`, `CompiledObservationPipeline`, `CompiledRewardEngine`, `CompiledTerminationEvaluator`) when `agent` is set; legacy engines otherwise. `set_agent()` recompiles atomically.
- `reset()`: domain randomization (scenario def or legacy), spawn jitter, scenario `spawn_override`, `episode_config` custom pose, `target_speed_override` applied to reward engine, `obstacle_overrides` instantiated as entities, `surface_friction_mult` applied.
- `step()`: action decode → vehicle physics → entity update → collision → track queries → checkpoints → reward → termination → `episode_config.max_duration_seconds` truncation → sensors → observation.

## 2. Current Training Flow

There are **two** training entry points, both ad-hoc:

1. **In-process**: `experiments/run_ui_env_ppo.py`, `run_ppo_experiment.py` — instantiate `SimulationEnvironment` directly, run `PPORunner` (PyTorch PPO in `sim_client/agents/ppo_baseline.py`), save `metrics.json` + `checkpoints/*.pt` + README. Verified working; `test_ppo_ui_env.py` covers template → export → compiled env → PPO (256 steps).
2. **TCP/external**: `main.py --headless --port N` hosts the env on `SimulationServer`; `sim_client/agents/ppo_train.py` (a simpler, older PPO) connects via `SimGymEnv` → `SimulationClient` → `HANDSHAKE`.

`TrainingExporter.export_training_bundle()` writes `environment.json`, `scenario.json`, `experiment.json`, and a generated `run_gym_training.py` that reconstructs the env in-process and runs `PPORunner`.

## 3. Current Experiment Flow

`ExperimentConfig` (`sim_env/experiment.py`) is a flat dataclass: algorithm, timesteps, rollout_steps, lr, gamma, gae_lambda, clip_coef, batch_size, epochs, eval/checkpoint frequency, output_directory. It is serialized with the project and exported — but it is **only a config bag**. There is no:

- experiment identity/fingerprint
- experiment lifecycle (create → launch → immutable)
- run entity or run lifecycle states
- run manager, orchestrator, metrics pipeline, evaluation system
- checkpoint registry, artifact management, batching, comparison, reproducibility check
- CLI for experiment operations

The two `experiments/baseline_*` directories are hand-produced outputs, not managed artifacts.

## 4. Existing APIs

| Surface | Status |
|---|---|
| `SimulationEnvironment.reset(seed, options)` / `step(action)` | Gymnasium 5-tuple semantics; rich `info` dict (termination_info, reward_breakdown, lateral_offset, heading_error, checkpoints) |
| `set_agent(agent)` | Atomic recompile of all four compiled pipelines |
| `set_road_definition(road_def)` | Regenerates mesh, queries, checkpoint tracker, broadphase |
| `EnvironmentValidator.validate(road_def, agent, entities)` | Gatekeeper, `is_valid_for_rl` |
| `EnvironmentVersionManager.compute_fingerprint(dict)` | SHA-256 over sorted JSON minus `version`/`timestamp`/`last_saved`/`author` |
| `EnvironmentVersionManager.compare_configurations(a, b)` | Subsystem diff — **already compares Phase 3 agent sub-fields** |
| `EnvironmentProject.save/load` | schema migration, `auto_increment` patch bump on fingerprint change |
| `TrainingExporter.export_training_bundle` | env/scenario/experiment JSON + runner script |
| `EpisodeRecorder` / `EpisodeReplayPlayer` | frame capture, JSON/gz save/load, scrub player |
| `SimulationServer` / `SimulationClient` / `SimGymEnv` | NDJSON TCP: HANDSHAKE, DISCOVER_CONTRACT, RESET, STEP, GET_STATE |
| `PPORunner` | full CleanRL-style PPO (GAE, clipped surrogate, metrics dict, `save_checkpoint`) |

## 5. Existing PPO Flow

`PPORunner(env, lr, gamma, gae_lambda, clip_coef, ..., num_steps, batch_size, epochs, seed)`:

- seeds torch + numpy; obs vector from compiled pipeline (23-dim default)
- rollout buffer (obs/actions/logprobs/rewards/dones/values), GAE, clipped surrogate + value clipping, entropy bonus
- metrics dict: episode returns/lengths, lateral error, speeds, collision/off-road counts, losses, KL, explained variance, SPS
- `save_checkpoint(path)`: torch.save model+optimizer+dims+seed+metrics

`ppo_train.py` is the older TCP variant (toy, no GAE update loop). `PPORunner` is the real baseline.

## 6. Existing Gym / TCP Integration

- `SimGymEnv` wraps `SimulationClient`; `HANDSHAKE` returns **legacy** `action_config`/`observation_schema` fields, not the Phase 3 contract. `DISCOVER_CONTRACT` returns the agent's exported schemas, reward graph, termination rules, sensors, scenario summary.
- Server is single-client, lockstep, non-blocking select loop polled by the app each frame. Headless mode polls in `run()` with `time.sleep(0.005)`.
- `PROTOCOL_VERSION = "2.0"` constant exists in `protocol.py` but server hardcodes the literal `'2.0'` twice and there is **no negotiation**: clients cannot declare supported versions.

## 7. Current Headless Capability

`python main.py --headless --port N` runs the full app without graphics. `SimulationEnvironment` itself has **zero** UI/render dependencies — it can be instantiated in any process (the PPO baseline already does this). The camera sensor has a procedural rasterizer fallback when no offscreen renderer is hooked. Multiple simulators are safe as **separate processes on separate ports**; multiple `SimulationEnvironment` instances in one process share no global state (verified: all state is per-instance).

## 8. Current Replay / Recording

- `EpisodeRecorder` captures frames (step, t, pos, yaw, speed, action, reward, breakdown, lat_offset, heading_err, collision) with metadata: timestamp, track_name, seed, env_version, scenario_name, env_fingerprint, env_config (full project dict), observation_schema, action_schema, reward_config.
- `app.py` **does** pass Phase 3 schemas + fingerprint when starting recording (verified, `app.py:533-547`).
- Recording only happens on the **manual-drive path** — externally driven (TCP) steps are not recorded.
- No trajectory dataset format (obs/action/reward/terminated/truncated per step for training data); recorder is for visual replay only.

## 9. Current Artifact / Metrics / Checkpoint Handling

- Metrics: `PPORunner.metrics` dict → `metrics.json` per baseline dir. Ad-hoc schema, not append-only, not scoped (step/episode/eval/run).
- Checkpoints: `*.pt` files dropped in `checkpoints/` with no registry/index.
- Artifacts: no registry, no run directory convention beyond the two ad-hoc baseline dirs.

## 10. Phase 3 Gap Re-Verification

| # | Phase 3 gap | Status now |
|---|---|---|
| P0-1 | Scenario overrides partially consumed | **MOSTLY FIXED.** Consumed: spawn, target_speed, obstacles, surface_friction, randomization (mass/tire/friction/spawn jitter). **Still dead:** `time_limit_override`, `sensor_noise_mult` (both the scenario field and the sampled `sensor_noise_mult` randomization param are dropped in `reset()`). |
| P0-2 | Inspector → runtime recompile hook | **FIXED.** `handle_property_change` fires `on_agent_modified` for `obs_/act_/rf_/term_/ag_` edits → `env.set_agent()` → atomic recompile. Scenario edits mutate the shared `scenario_def` object and apply on next reset. No explicit dirty-flag/validate→compile cycle but the mechanism is deterministic. |
| P1-3 | Config diff compares legacy keys | **FIXED.** `compare_configurations` diffs `agent.observation_space/action_space/reward_function/termination_rules` plus `scenario_def` alongside legacy keys. |
| P1-4 | Version doesn't react to config changes | **MOSTLY FIXED.** `EnvironmentProject.save(auto_increment=True)` bumps patch when fingerprint changed since last save/load. Fingerprint covers the full Phase 3 project dict. Caveat: fingerprint input includes `environment_version`, `schema_version`, `fingerprint` keys — version bumps pollute identity (see hardening §6.3). |
| P1-5 | Replay Phase 3 schema capture | **FIXED for UI path.** Missing only explicit `simulator_version`, `protocol_version`, and full `scenario` config (name only; `env_config` does embed the whole project). |
| P1-6 | PPO end-to-end with UI env | **VERIFIED.** `test_ppo_ui_env.py` + `experiments/baseline_ui_env/` artifacts. |
| P2-7 | Missing docs | **FIXED.** `ENVIRONMENT_VERSIONING.md`, `TRAINING_EXPORT.md` exist. |
| P2-8 | Protocol version hardcoded "2.0" | **OPEN.** No negotiation; clients can't declare supported versions. |

**Architectural risk — dual paths:** legacy and Phase 3 paths both live; all 90 tests pass across both. Phase 4 keeps both (external TCP clients on legacy handshake still function); new experiment tooling always snapshots the **full** project dict so identity is path-agnostic.

## 11. Architectural Risks for Phase 4

1. **HANDSHAKE returns legacy schemas** even when a Phase 3 agent is active — TCP trainers see `action_config`, not the agent's action space. Phase 4 contract must use `DISCOVER_CONTRACT`-equivalent data (in-process trainers get it from the project file directly).
2. **`_strip_metadata` incomplete**: `environment_version`, `schema_version`, `fingerprint` are hashed into the fingerprint — version bumps change identity without config change. Fix: strip them.
3. **No run-directory convention** — must establish one (spec §14) without breaking `experiments/baseline_*`.
4. **Subprocess safety**: orchestrator must launch `sys.executable -m <trainer>` with arg lists (no shell), validate the run dir path stays under the experiments root, capture stdout/stderr to files.
5. **UI is single-threaded pygame** — live run monitoring must poll `RunManager`/tail `metrics.jsonl` on a throttle, never block the frame loop.
6. **Global RNG use**: `PPORunner` seeds `np.random.seed()`/`torch.manual_seed()` globally — fine inside a dedicated trainer subprocess; do NOT run PPORunner in the studio process.

## 12. Recommended Implementation Order

```
Stage A (Phase 3 hardening — small, testable):
  A1. Consume time_limit_override (truncation) + sensor_noise_mult (scale sensor noise_std) in env
  A2. Strip environment_version/schema_version/fingerprint from fingerprint input
  A3. Protocol version negotiation (HANDSHAKE accepts supported_versions; server picks/errors)
  A4. Recorder metadata: simulator_version, protocol_version, scenario config

Stage B (Experiment core — pure domain, no I/O risk):
  B1. sim_experiment package: TrainingConfig/EvaluationConfig/ExperimentManifest + fingerprints
  B2. ExperimentManager (create/validate/save/load/clone/list/immutability/export/archive)
  B3. Run + RunManager (lifecycle states, run dir layout, run.json)
  B4. MetricsWriter (buffered JSONL) + MetricsReader + scoped aggregation
  B5. ArtifactRegistry (checkpoints/eval/replays/trajectories index JSONL)

Stage C (Orchestration):
  C1. Trainer contract v1.0 (contract.json into run dir; metrics.jsonl/checkpoints/evals/run_result out)
  C2. LocalTrainingOrchestrator (subprocess, logs, poll, cancel, timeout, failure states)
  C3. Contract-compliant PPO trainer entry point (sim_experiment/trainers/ppo_trainer.py)
  C4. Resume-from-checkpoint (new run referencing parent run + checkpoint)

Stage D (Supporting systems):
  D1. EvaluationRunner (headless eval of checkpoint artifacts → evaluation results)
  D2. TrajectoryWriter (agent_data vs diagnostic_data separation)
  D3. Batch expansion (seeds, scenarios)
  D4. ReproducibilityChecker
  D5. Headless worker pool + benchmark script

Stage E (Surfaces):
  E1. CLI (sim_experiment/cli.py): validate/create/list/launch/runs/inspect/cancel/evaluate/export/benchmark/resume
  E2. UI: TRAIN tab in inspector (experiment list, run monitor, sparkline charts, launch/cancel) — calls services only

Stage F (Validation):
  F1. New test files (manifest, runs, metrics, eval, artifacts, reproduce, batch, headless, protocol, PPO smoke)
  F2. Performance benchmark → docs/PHASE_4_PERFORMANCE.md (measured numbers only)
  F3. Docs (12 files) + PHASE_4_FINAL_REPORT.md
```
