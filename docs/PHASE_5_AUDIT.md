---
noteId: "feb9e9c0bf0311f1a29f1fbaabbd87c8"
tags: []

---

# PHASE 5 AUDIT — Verified Repository State

Date: 2026-10-03 · Verified against HEAD `2589b28` · Test suite: **148/148 passing** (33.9s)

This audit verifies the Phase 4 final report against the actual code before
any Phase 5 work begins. Where the report and the code disagree, the code
wins and the discrepancy is recorded below.

---

## 1. Verified Baseline

| Check | Result |
|---|---|
| Git HEAD | `2589b28` "feat: Add trajectory storage and reading functionality" |
| Working tree | clean (no uncommitted changes) |
| Test suite | `python -m pytest tests -x -q` → **148 passed**, 1 pygame warning |
| Package layout | `sim_core`, `sim_env`, `sim_experiment` (+`trainers/`), `sim_net`, `sim_client` (+`agents/`), `sim_recorder`, `sim_project`, `sim_render`, `sim_ui`, `tests` (30 files), `benchmarks`, `presets`, `docs` (37 files) |
| Real experiment on disk | `experiments/exp_556fe0488d4db907/` with run artifacts, checkpoints, evals, replays |

Phase 4 claims verified functional (not just declarative):

- **Orchestrator** — `LocalTrainingOrchestrator.launch/poll/wait/cancel` spawns
  whitelisted `sim_experiment.trainers.*` subprocesses, syncs metrics/artifacts
  into `run.json`, enforces `max_wall_seconds`, terminate→kill cancel.
- **RunManager** — validated state machine `CREATED→QUEUED→STARTING→RUNNING→
  {PAUSED,COMPLETED,FAILED,CANCELLED,INTERRUPTED}`; terminal states final;
  atomic `run.json` writes.
- **Contract** — `contract_version "1.0"`, `build_contract`/`validate_contract`,
  env_mode inprocess|tcp, resume block, path materialization.
- **PPO trainer** — real external process (`ppo_trainer.py`), consumes contract,
  builds envs via `build_env_from_dicts` or `SimGymEnv` (tcp), drives
  `PPORunner`, writes metrics/checkpoints/evals/replays/trajectories.
- **Metrics** — scoped buffered JSONL (`step|episode|evaluation|run`), NaN-safe,
  seq continues across restart, torn-tail tolerant reader.
- **Evaluation** — `evaluate_policy` frozen-policy/fresh-env/fixed-seeds;
  `make_policy_from_checkpoint` PPO adapter only.
- **Artifacts** — append-only `registry.jsonl`, path-traversal safe.
- **Trajectories** — per-episode JSONL, `agent_data`/`diagnostic_data` split
  enforced by required keys.
- **Headless** — `build_env_from_dicts`, `HeadlessEnvPool` (in-proc, seed+i),
  `HeadlessSimProcessPool` (`main.py --headless --port`).
- **Batch expansion** — `expand_run_specs` deterministic seeds×scenarios.
- **Reproducibility** — `check_reproducibility` 12 checks, tamper detection.
- **CLI** — 15 subcommands over services only, no business logic.
- **UI TRAIN tab** — provider/action hook wiring verified (`inspector.py:250-292`,
  `app.py:320-474`); visuals not manually confirmed (unchanged from Phase 4).

---

## 2. Corrections to the Phase 4 Report (stale items)

The audit found three Phase 4 "limitations" that do not match current code:

1. **"`obs_leak` detection in `get_state` uses heuristic name matching"** —
   STALE. The `GET_STATE` handler (`sim_net/server.py:253-262`) returns a fixed
   payload (speed/pos/yaw/sim_time/total_reward/reward_breakdown) with no name
   matching. The real leak guard is already structural:
   `ObservationChannelConfig.category` ∈ {`agent_observation`,
   `debug_telemetry`, `oracle_ground_truth`}, enforced at pipeline compile time
   (`observation_designer.py:307-315`) and re-checked by `EnvironmentValidator`
   (`validator.py:176-180`). The only name-substring heuristic is a benign
   default-fill (`observation_designer.py:360`, `"lidar" in c.name`).

2. **"Hardcoded `['lidar_rays','rgb_camera','imu']` in `get_state`"** —
   STALE (wrong location). The actual hardcoded sensor-name dependencies are:
   - `AgentDefinition.sensor_names` default `["vehicle_state","lidar_rays",
     "rgb_camera","imu"]` (`agent.py:60,75,110`) — a binding default, not a
     contract.
   - `EnvironmentValidator` fallback `available_sensors or [...]` same list
     (`validator.py:98`) — silently assumes sensors that may not exist.
   - Legacy `_build_observation` path (`environment.py:520-590`) hardcodes
     `vehicle_state`/`lidar_rays`/`rgb_camera` lookups for the OLD
     `ObservationSchema` (used only when no `agent`/`compiled_obs_pipeline`).
   - UI defaults `inspector.py:93`, `app.py:263,778` (camera PiP lookup).

3. **"`ep_len` counters exist but unused in single-env episode stats"** —
   STALE. PPO's per-env `ep_len` IS consumed into `ep_stats['length']` →
   episode-scope metrics for any `num_envs`. The real defects are:
   - **Three decoupled counters**: `env.current_step` (environment.py:121,335),
     PPO `ep_len[]` (ppo_baseline.py:192-286), eval `ep_len`
     (evaluation.py:79-108). They diverge when `step()` is called post-done
     (env early-returns at environment.py:324-333 without incrementing).
   - **Unseeded mid-episode reset**: `env.reset()` at ppo_baseline.py:289 drops
     the seed after the first episode — breaks seed independence across
     episodes (only first reset is seeded `seed+e`).
   - **Truncation treated as terminal in GAE**: `done = terminated or truncated`
     (ppo_baseline.py:237) means no V(s') bootstrap at time-limit truncation —
     standard CleanRL caveat, biases values on truncated episodes.
   - **Action clamp mismatch**: the unclamped sampled action is stored in the
     rollout buffer while the env receives the clamped action
     (ppo_baseline.py:227-232).
   - **`act_dim` hardcoded to 3** (ppo_baseline.py:135) — not derived from
     `action_space`/schema.

---

## 3. Current State by Subsystem

| Subsystem | State | Notes |
|---|---|---|
| CurriculumDefinition | **Declarative only** | `sim_env/curriculum.py` — 5 default stages with `scenario_id`, `target_metric`, `advancement_threshold`, `min_episodes`, `environment_overrides`. Serialized into manifest (`curriculum_configuration`) but **never consumed at runtime**. No `set_scenario` API — scenario swap = assign `env.scenario_def` + `reset()` (UI does this at app.py:181/204/237). |
| Batch | **Partial** | `expand_run_specs` + `cli batch` create CREATED runs; nothing dispatches them. No queue, no workers, no retry, no `batch_result`. |
| Trainer contract | **Functional, no capabilities** | `capabilities` field exists but is a static list of artifact kinds; no algorithm/action-space/obs-type declaration; no pre-launch compat check (ppo_trainer fails at runtime with `unsupported_algorithm`). |
| Algorithms | **PPO only** | `ppo` + `dummy` in `TRAINER_MODULES`. `algorithm` string validated only inside the trainer process. |
| Evaluation | **Functional** | Algorithm-agnostic `act_fn`; checkpoint adapter PPO-only. Not consumed by any runtime loop for decisions. |
| Trajectories | **Functional, no explorer/export** | Writer+reader exist; `cli trajectories` only lists files. No dataset builder, no filtering, no schema-compat check. |
| Comparison/analytics | **Missing** | MetricsReader can aggregate; nothing aligns/compares runs. |
| Remote workers | **Missing** | `sim_net` serves the *simulator* env protocol (RESET/STEP/GET_STATE), not a worker/scheduler protocol. |
| Observation security | **Mostly structural** | Category system real and enforced at compile time; gaps: validator sensor fallback, legacy `_build_observation` hardcoded names, `export_schema` includes category but consumers don't act on it, `info` dict contents unclassified. |
| Legacy dual paths | **Present** | `SimulationEnvironment` keeps legacy `scenario_config`/`observation_schema`/`action_config`/`reward_config`/`termination_config` alongside Phase-3 `agent`+`scenario_def`. Every consumer prefers `scenario_def`/`compiled_*` when present. |
| ep_len | **Partially defective** | See §2 item 3. |

---

## 4. Architectural Risks

1. **Curriculum needs a scenario-swap seam.** Env consumes `scenario_def` only
   inside `reset()`; entities spawned by a scenario persist. A safe swap needs
   `set_scenario(scenario_def)` that clears scenario entities + resets —
   small, additive, keeps `sim_core` clean.
2. **Trainer↔runner coupling.** `ppo_trainer.py` duplicates env-build, policy
   adapter, eval+replay wiring, trajectory capture. SAC/DQN will share ~60%
   of this — extract a shared trainer harness (`trainers/_common.py`) rather
   than forking three copies.
3. **act_dim/action-shape assumptions** in PPORunner and `_policy` adapters
   hardcode `[steer,throttle,brake]` semantics; SAC needs Box-bounds-aware
   squashed-Gaussian output, DQN needs Discrete. Capability declarations must
   carry action-space type + dims so validation happens before launch.
4. **Run status ownership.** Orchestrator is sole writer of `run.json` in
   Phase 4. A BatchScheduler dispatching through the same orchestrator
   preserves that; a parallel writer would break invariants — workers must
   delegate lifecycle writes to the orchestrator/run-manager APIs.
5. **Checkpoint compatibility surface.** PPO `.pt` contains
   `weights_only=False` pickled dicts — adding curriculum state is safe but
   must be version-tolerant on load (missing key → fresh curriculum state
   only when contract says so).
6. **Windows.** No `fork`; scheduler workers are `subprocess.Popen` (already
   the pattern). `multiprocessing` would need `spawn` — keep subprocess.
7. **Immutability.** Batch resume creates NEW runs — good; scheduler must
   never relaunch in-place. Retries create new runs linked via `resume_from`
   or fresh seeds — must record audit trail in `batch_result.json`.

---

## 5. Recommended Implementation Order (adjusted from prompt)

The prompt order stands; repository evidence adds two refinements:

1. **Trainer capability declaration FIRST (before SAC/DQN)** — it is a small
   contract extension (`capabilities` block + `check_compatibility` in the
   orchestrator launch path) that P0 items don't depend on, and it prevents
   P1 work from re-failing at runtime.
2. **Shared trainer harness extraction as part of SAC** — `_build_envs`,
   metrics/registry/trajectory/eval+replay wiring, `_write_result`,
   resume-checkpoint resolution. Extracted once, consumed by SAC and DQN.
3. P0 sequence: curriculum runtime → curriculum persistence/eval →
   observation-contract hardening (small, now that real gaps are known) →
   ep_len fix → batch scheduler + local worker pool.
4. P1: capability discovery → shared harness → SAC → DQN → comparison/
   analytics → trajectory/dataset export → PPO multi-env edge tests.
5. P2: remote workers (LAN, authenticated, capability-declared) → legacy
   migration analysis → UI → CLI → benchmarks → manual validation → docs.

`get_state` rework is NOT needed (handler is already a fixed diagnostic
payload). The sensor-name work is: validator fallback removal, legacy
`_build_observation` documentation/retention decision, and exposing the
category field to consumers (trajectory/dataset metadata).

---

## 6. What Is NOT in the Repo

- No scheduler/worker abstraction of any kind.
- No replay buffer / off-policy infrastructure anywhere.
- No discrete-action trainer path exercised (env supports it; untested).
- No dataset format, no imitation-learning surface.
- No remote worker protocol (sim TCP is env-only, single-client).
- No `set_scenario` env API.
- No run comparison, no chart-series service beyond sparkline data.
