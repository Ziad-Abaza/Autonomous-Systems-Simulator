---
noteId: "b1a594c0bf0011f1a29f1fbaabbd87c8"
tags: []

---

# PHASE 4 FINAL REPORT — Training & Experiment Platform

Date: 2026-10-03 · Simulator version: 4.0.0 · Protocol: 2.0 (negotiated)

## 1. Executive Summary

Phase 4 delivered a complete, reproducible experiment platform on top of
the Phase 3 RL environment architecture: immutable experiment manifests
with deterministic identity, a run manager with validated lifecycle
states, a local subprocess orchestrator with a versioned external-trainer
contract, structured metrics, separate evaluation, checkpoint/artifact
tracking, run-linked replays and trajectories, deterministic batch
expansion, headless + parallel environments, a CLI, and a UI experiments
panel.

**The training algorithm lives outside the simulation core** — PPO runs
in `sim_experiment.trainers.ppo_trainer` (subprocess); `sim_core/` and
`sim_env/` contain no training loop.

Test suite: **148/148 passing** (was 87 at Phase 3 audit). A real
512-step external PPO run was launched, completed, checkpointed,
evaluated, replayed, verified reproducible, and exported — see §30.

## 2. Initial Repository State

Baseline before Phase 4 work (verified, not assumed):

- `87/87` tests passing (Phase 3 audit figure).
- Existing modules: `sim_core`, `sim_env` (agent/obs/action/reward/
  termination/scenario/randomization/curriculum/validator/versioning/
  export), `sim_net` (TCP 2.0, no negotiation), `sim_client`
  (PPO baseline), `sim_recorder`, `sim_project`, `sim_ui`.
- `experiments/` contained ad-hoc PPO artifacts (baseline_ppo,
  baseline_ui_env) — no experiment/run abstraction.
- No orchestration, run lifecycle, metrics pipeline, evaluation system,
  artifact registry, batch, reproduce, headless pool, or platform CLI.

## 3. Phase 3 Gaps Verified

From `PHASE_3_AUDIT.md` — verified against actual code:

| Claimed gap | Verified state |
|---|---|
| Scenario overrides partially consumed | **Real** — `time_limit_override` + `sensor_noise_mult` unused |
| Inspector→runtime recompilation | **Already done** (`on_agent_modified` → `set_agent`) |
| Config diff legacy-only | **Already done** (Phase 3 agent fields compared) |
| Version doesn't react to changes | **Already done** (auto-increment on save); new issue found: fingerprint included version fields |
| Replay lacks Phase 3 schemas | **Already done** (schemas recorded); version fields missing |
| PPO from UI-created env | **Already done** (`test_ppo_ui_env`, run scripts) |
| ENV_VERSIONING / TRAINING_EXPORT docs | **Already existed** |
| No protocol negotiation | **Real** — hardcoded `"2.0"` |
| Legacy+Phase3 dual paths | **Real** — kept, documented as debt |

## 4. Phase 3 Hardening Performed

Test-first (`tests/test_phase4_hardening.py`, 12 tests, RED→GREEN):

1. `time_limit_override` consumed — truncates with
   `scenario_time_limit_exceeded` (env step, environment.py).
2. `sensor_noise_mult` (scenario × randomization) applied per-reset to
   base sensor noise — idempotent, non-compounding.
3. `compute_fingerprint` strips `environment_version`/`schema_version`/
   `fingerprint` — version bumps no longer alter identity.
4. TCP version negotiation — clients declare `protocol_versions`;
   server picks highest mutual or `protocol_version_mismatch` error;
   `SUPPORTED_PROTOCOL_VERSIONS` exported.
5. Recorder metadata — `simulator_version`, `protocol_version`,
   `scenario_config` captured; UI passes the scenario dict.
6. `sim_version.py` — single SIMULATOR_VERSION source of truth.

## 5. Experiment Architecture

New package `sim_experiment/` (domain layer) + `sim_experiment/trainers/`
(external processes). The UI (`sim_ui`) and CLI call services; only the
orchestrator spawns trainers; only the orchestrator writes `run.json`;
trainers report through contract artifacts. Full layout in
`PHASE_4_OVERVIEW.md`.

## 6. Experiment Manifest

`ExperimentManifest.from_project()` snapshots: full environment dict
(not a reference), scenario config incl. randomization, obs/action
schemas, reward/termination/episode/curriculum configs, training +
evaluation configs, seed, simulator/protocol versions. `manifest_version
= "1.0"`. Immutable after `mark_launched()` — enforced by
`ExperimentManager.create/save`.

## 7. Experiment Identity

`experiment_fingerprint` = SHA-256 canonical-JSON over {env fingerprint,
scenario, seed, schemas, all configs, training, evaluation, simulator
version, protocol version}. `experiment_id = exp_<16hex>`. Name and
timestamps excluded. Verified: identical config → identical fingerprint;
different seed/algorithm/config → different id (tests).

## 8. Run Manager

`Run`/`RunManager` with validated transitions
(`CREATED→QUEUED→STARTING→RUNNING→{PAUSED,COMPLETED,FAILED,CANCELLED,
INTERRUPTED}`). Invalid transitions raise; terminal states are final.
`run.json` tracks pid, progress, latest metrics, artifact refs, error,
exit code, `resume_from`. Atomic writes.

## 9. Training Orchestration

`LocalTrainingOrchestrator.launch()` → contract → subprocess
(`python -m sim_experiment.trainers.*`, arg list, no shell). `poll()`
syncs metrics/artifacts + detects exit + enforces `max_wall_seconds`
(→INTERRUPTED). `cancel()` → terminate→kill→CANCELLED. `wait()` polls to
terminal. `resume` creates a NEW run linked via `resume_from`. Verified:
completed run, simulated crash → FAILED, live cancel → CANCELLED,
whitelist enforcement (all tested).

## 10. External Trainer Contract

`contract_version "1.0"`: experiment/env identity, seed, versions,
schemas, training+evaluation configs, env_mode + tcp ports, paths,
resume. `validate_contract()` gates launch. Trainer outputs:
`metrics.jsonl`, `checkpoints/` + registry, `evaluation/`,
`trajectories/`, `replays/`, `run_result.json`. Trainers restricted to
`sim_experiment.trainers.*` namespace.

## 11. Metrics System

`metrics.jsonl` — scoped records `{scope,seq,ts,timestep,metrics}`.
Buffered `MetricsWriter` (64-record default, seq continues across
restart); `MetricsReader` skips torn tail lines, provides
by_scope/latest/tail/aggregate. NaN/Inf → `null` (tested).
Measured overhead: **−0.17%** (noise) vs. no metrics.

## 12. Evaluation System

`evaluate_policy(env, act_fn, seeds, num_episodes, step_observer)` —
frozen policy + fresh env + fixed seeds → `EvaluationResult` with
per-episode stats and aggregates (reward/length/completion/collision/
off-road/timeout rates, lat/heading errors, speed). Runs: in-training
at `eval_frequency` (deterministic actor-mean), via CLI `evaluate`, via
UI EVALUATE. Persisted as `evaluation/eval_<step>.json` + registry.

## 13. Checkpoint/Artifact System

`ArtifactRegistry` (`artifacts/registry.jsonl`) tracks
checkpoint/evaluation/replay/trajectory references with step + metadata
(experiment_id, run_id, env_fingerprint, algorithm). Path traversal and
missing files rejected. PPO saves `policy_<step>.pt` at
`checkpoint_frequency` + `policy_final.pt` (model+optimizer+timestep →
resumeable). `poll()` syncs refs into `run.json`.

## 14. Replay Integration

Recorder metadata now captures simulator/protocol version + scenario
config + Phase 3 schemas + fingerprint. PPO trainer writes
recorder-compatible `replays/eval_<step>_ep0.json` for the first eval
episode and registers it; `run.json.replays` tracks all linked replays.
UI replay player loads the same frame format.

## 15. Trajectory/Dataset System

`TrajectoryWriter` — per-episode JSONL: header (env fingerprint,
scenario, seed, schemas) + step records with **`agent_data`** (obs,
action, reward, flags, reason) vs **`diagnostic_data`** (speed,
offsets, collision) strictly separated. Buffered; NaN/NumPy sanitized;
required keys enforced. Enabled via `algorithm_config.
trajectory_episodes`; `TrajectoryReader` for inspection.

## 16. Batch Experiments

`expand_run_specs(manifest, seeds, scenario_ids)` — deterministic
cross product; `cli batch` materializes CREATED runs. Manifest identity
unchanged (runs vary, not the experiment).

## 17. Headless Execution

`main.py --headless` (no graphics init, TCP 60 Hz loop) — verified in
CI tests. `build_env_from_dicts()` factory: serialized dicts → env, zero
UI/render deps (tested). `HeadlessSimProcessPool` spawns N simulator
processes with port wait + cleanup (TCP env_mode).

## 18. Parallel Environments

In-process: `HeadlessEnvPool` N independent envs (seed+i; tested
independence). PPORunner accepts env lists — contiguous per-env rollout
segments with per-segment GAE (correct bootstrapping). TCP: N headless
sim processes via `SimGymEnv`. Independent seeds, episodes, metrics.

## 19. Reproducibility

`check_reproducibility(exp_dir)`: 12 checks incl. env-fingerprint
recompute vs recorded, scenario equality, schema presence, version
compatibility. Fatal vs warning failures; tampered env → detected
(tested). `exact_reproduction_guaranteed` always false by design
(torch nondeterminism/hardware) — honest reporting.

## 20. CLI

`python -m sim_experiment.cli`: validate-env, create, list, show,
launch (--wait/--env-mode), batch, runs, status (--watch), cancel,
resume, evaluate, trajectories, reproduce, export, archive, benchmark.
All verified against the real `experiments/` root (see §30).

## 21. UI

Inspector `TRAIN` tab (RL category): experiment list + CREATE FROM ENV,
LAUNCH PPO / CANCEL / RESUME / EVALUATE / VERIFY / EXPORT actions, run
monitor (status, timesteps, episodes, latest metrics, reward sparkline
via new `chart` property type, error display). Services wired in app.py
(`_train_provider`, `_train_action`, `_poll_training_runs` ~1 Hz).
UI calls services only — no direct file/process manipulation.
**Manual validation of interactive panel requires running the app** —
the code path is implemented and import-verified; I cannot confirm
visual behavior without a live session.

## 22. PPO End-to-End Validation

Real run (this session, repo `experiments/`):

- `create` → `exp_556fe0488d4db907` (oval preset, 512 steps, ckpt/eval
  every 256)
- `launch --trainer ppo` → `run_20261003_105952_ebf6d909` → COMPLETED
- Artifacts: `policy_256.pt`, `policy_512.pt`, `policy_final.pt`;
  `eval_256.json`, `eval_512.json`; `eval_256_ep0.json`,
  `eval_512_ep0.json` replays; `metrics.jsonl`; `contract.json`; logs.
- `evaluate` (policy_final, 5 episodes): mean_reward 1185.85,
  timeout_rate 1.0, mean_speed 0.0033 — honest output of a 512-step
  untrained policy.
- `reproduce` → `reproducible: true`. `export` → full bundle copied.
- Automated equivalent: `TestPPOEndToEnd` (subprocess → contract →
  metrics → checkpoint → headless eval).

## 23. Tests

| Suite | Tests | Result |
|---|---|---|
| Phase 3 hardening (`test_phase4_hardening.py`) | 12 | PASS |
| Experiment domain (`test_experiment_domain.py`) | 30 | PASS |
| Platform integration (`test_training_platform.py`) | 16 | PASS |
| Pre-existing Phase 1–3 | 90 | PASS |
| **Total** | **148** | **148/148 PASS in ~33s** |

New tests cover: manifest identity/immutability, manager CRUD/export/
archive, run transitions/resume, metrics buffering+NaN, artifact safety,
contract validation, orchestrator launch/failure/cancel/whitelist,
evaluation aggregates, trajectory separation, batch expansion,
reproducibility detection, headless independence, PPO e2e.

## 24. Performance Benchmarks

`benchmarks/benchmark_phase4.py` → `benchmarks/phase4_results.json`:

| Metric | Result |
|---|---|
| 1 env | 352.0 env-steps/s (2.84 ms) |
| 2 envs | 340.3 agg (170.2/env) |
| 4 envs | 332.4 agg (83.1/env) |
| sensors: full / lidar+state / state | 341.9 / 364.6 / 1393.8 sps |
| metrics overhead | −0.17% (noise) |
| contract build+validate | 0.05 ms |

Interpretation in `PHASE_4_PERFORMANCE.md`: sensors dominate step cost;
in-process parallelism adds diversity not wall-clock speed.

## 25. Security

- Trainer modules whitelisted to `sim_experiment.trainers.*`; subprocess
  as argv list (no shell strings, no eval/exec).
- Run dirs confined to experiments root; artifact paths can't traverse
  run dir; episode ids can't traverse trajectory dir.
- JSONL/JSON only — no pickle/yaml/network in the persistence layer.
- Checkpoint `torch.load(weights_only=False)` on **platform-written**
  files only; never loaded from arbitrary UI/network input.
- Logs captured per run; failures never hidden (run_result + run.json
  error records).

## 26. Documentation

Created: `PHASE_4_AUDIT.md`, `PHASE_4_PLAN.md`, `PHASE_4_OVERVIEW.md`,
`EXPERIMENT_MANIFEST.md`, `EXPERIMENT_IDENTITY.md`, `RUN_MANAGER.md`,
`TRAINING_ORCHESTRATION.md`, `TRAINER_CONTRACT.md`, `METRICS_SCHEMA.md`,
`EVALUATION_SYSTEM.md`, `CHECKPOINT_ARTIFACTS.md`, `REPLAY_TRAJECTORY.md`,
`BATCH_EXPERIMENTS.md`, `HEADLESS_ENVIRONMENTS.md`,
`PARALLEL_ENVIRONMENTS.md`, `EXPERIMENT_REPRODUCIBILITY.md`,
`CLI_REFERENCE.md`, `PHASE_4_PERFORMANCE.md`, this report.

## 27. Known Limitations

- Single-client TCP server; TCP env_mode spawns separate sim processes.
- In-process parallel envs are CPU-bound (GIL) — diversity, not speed.
- `cli batch` creates CREATED runs; orchestrating all batch runs is a
  manual launch step (no batch scheduler).
- Only PPO has a trainer + checkpoint adapter; SAC/DQN named but not
  implemented.
- Bitwise reproducibility not guaranteed (torch/BLAS/HW) — reported
  honestly.
- UI TRAIN panel not manually exercised in a live GUI session.
- Multi-env contiguous segments reduce per-env rollout length; very
  short rollouts may degrade PPO quality.

## 28. Remaining Technical Debt

- Legacy + Phase 3 dual paths in `SimulationEnvironment` (unchanged —
  documented risk; removal is Phase 5 scope).
- `obs_leak` detection in `get_state` endpoint uses heuristic name
  matching only.
- Hardcoded `['lidar_rays','rgb_camera','imu']` in `get_state`.
- `ep_len` counters exist but unused in single-env episode stats.
- PPO multi-env GAE uses per-segment bootstrapping — correct, but
  env-count-dependent rollout shapes.

## 29. Recommended Phase 5

1. Curriculum advancement during training runs (hooks exist; stages not
   consumed by trainer loop).
2. Batch scheduler (parallel run queue + worker pool).
3. SAC/DQN trainers on the contract (adapter seam is in place).
4. Experiment comparison UI (metrics overlay, eval diff table).
5. Trajectory replay viewer + dataset export (imitation learning).
6. Remote/TCP trainer workers + distributed runs.
7. Legacy code-path removal once Phase 3 path is fully canonical.

## 30. Exact Verification Commands / Results

```powershell
# Full suite
python -m pytest tests -x -q
# 148 passed, 1 warning in 33.34s

# Hardening tests
python -m pytest tests/test_phase4_hardening.py -x -q   # 12 passed
python -m pytest tests/test_experiment_domain.py -x -q  # 30 passed
python -m pytest tests/test_training_platform.py -x -q  # 16 passed (incl. PPO e2e)

# Real acceptance workflow (repo experiments/ root)
python -m sim_experiment.cli create --project presets/oval_circuit.sim.json `
    --scenario basic_lane_following --name phase4_validation `
    --timesteps 512 --rollout 256 --batch-size 128 --ckpt-freq 256 --eval-freq 256 --seed 42
#   -> exp_556fe0488d4db907
python -m sim_experiment.cli launch exp_556fe0488d4db907 --trainer ppo --wait 120
#   -> run_20261003_105952_ebf6d909, status COMPLETED, timesteps 512
python -m sim_experiment.cli evaluate exp_556fe0488d4db907 run_20261003_105952_ebf6d909
#   -> mean_reward 1185.8451, timeout_rate 1.0 (untrained, 5 eps)
python -m sim_experiment.cli reproduce exp_556fe0488d4db907   # -> reproducible: true
python -m sim_experiment.cli export exp_556fe0488d4db907 --dest experiments\exported\exp_556fe0488d4db907
python -m sim_experiment.cli runs exp_556fe0488d4db907        # -> COMPLETED, 512 steps

# Benchmarks
python benchmarks/benchmark_phase4.py   # -> benchmarks/phase4_results.json

# CLI round-trip on temp root (dummy trainer)
python -m sim_experiment.cli --root %TMP%\phase4cli launch <exp> --trainer dummy --wait 30
#   -> status COMPLETED
```

**Files created:** `sim_version.py`, `sim_experiment/` (manifest,
manager, run, metrics, artifacts, trainer_contract, orchestrator,
evaluation, trajectory, batch, reproduce, headless, cli, trainers/×3),
`tests/test_phase4_hardening.py`, `tests/test_experiment_domain.py`,
`tests/test_training_platform.py`, `benchmarks/benchmark_phase4.py`,
`benchmarks/phase4_results.json`, 19 `docs/PHASE_4*`/`docs/*.md` files.

**Files modified:** `sim_env/environment.py` (scenario overrides),
`sim_env/versioning.py` (fingerprint strip), `sim_net/protocol.py` +
`server.py` (negotiation), `sim_client/client.py` (declared versions),
`sim_client/agents/ppo_baseline.py` (multi-env, hooks, resume),
`sim_recorder/recorder.py` (metadata), `sim_ui/inspector.py` (TRAIN tab +
chart), `sim_ui/app.py` (services, monitor).

**Claims I cannot confirm:** interactive TRAIN-panel visuals (no live GUI
session run), multi-day-stability of long trainings, bitwise
reproducibility across machines.
