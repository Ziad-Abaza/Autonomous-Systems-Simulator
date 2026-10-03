---
noteId: "5a0dc280bf0c11f1a29f1fbaabbd87c8"
tags: []

---

# PHASE 6 AUDIT — Verified Baseline

Audit performed: 2026-10-03, against git HEAD `2f6a518` (last Phase 5 commit).
Method: direct repository inspection + full test suite run. The Phase 5
final report was treated as historical context and every claim below was
checked against the code.

## 1. Baseline test run

```
python -m pytest tests -x -q
→ 252 passed, 1 warning (pygame pkg_resources deprecation) in ~256 s
```

**The Phase 5 claim "252/252 tests passing" is VERIFIED.**

## 2. What Phase 5 actually implemented (verified in code)

| Claim | Status | Evidence |
|---|---|---|
| Curriculum runtime + persistence + resume | VERIFIED | `sim_experiment/curriculum_runtime.py` (CurriculumController, fingerprint, to_state/from_state); `curriculum_state.json` per run; checkpoint-embedded state |
| Batch scheduler + local worker pool | VERIFIED | `sim_experiment/scheduler.py` — jobs, attempts, retries, `batch.json`/`batch_result.json`, LocalAdapter |
| PPO, SAC, DQN trainers | VERIFIED | `sim_experiment/trainers/{ppo,sac,dqn}_trainer.py` + `sim_client/agents/{ppo,sac,dqn}_baseline.py` sharing `_harness.py` |
| Trainer capability validation | VERIFIED | `capabilities.py` TRAINER_CAPABILITIES + `check_compatibility` pre-launch |
| Analytics + comparison | VERIFIED | `analytics.py` compare_runs/compare_experiments/metric_summary |
| Trajectory explorer | VERIFIED | `trajectory.py` TrajectoryWriter/Reader, `dataset.list_episodes` |
| transitions_v1 export | VERIFIED | `dataset.py` — manifest.json + episodes.jsonl, agent_data-only steps |
| Observation-contract security | VERIFIED | `observation_contract.py` diagnostic contract; dataset export structurally excludes diagnostics |
| Authenticated LAN remote workers | VERIFIED | `remote_worker.py` — NDJSON, hmac.compare_digest token, version gate "1.0", HELLO/LAUNCH/POLL/CANCEL |
| PPO multi-env correctness fixes | VERIFIED | `ppo_baseline.py` terms_buf/final_values_buf truncation bootstrap, `_next_reset_seed` |
| CLI + TRAIN UI | VERIFIED | `cli.py` 21 commands; TRAIN tab in `inspector.py` via provider/action pattern |
| Measured Phase 5 performance | VERIFIED | `benchmarks/phase5_benchmarks.py` + `docs/PHASE_5_PERFORMANCE_RAW.json` exist |

## 3. Environment stepping architecture (current reality)

- All envs satisfy a duck type: `reset(seed, options)->(obs,info)`,
  `step(action)->(obs,reward,terminated,truncated,info)`, `close()`.
- `build_envs_from_contract` (`_harness.py:37`) returns a **plain Python
  list**: `SimulationEnvironment` objects (inprocess) or `SimGymEnv`
  TCP clients (tcp).
- **PPO** (`ppo_baseline.py:258`): contiguous per-env rollout segments —
  `env_idx = step // steps_per_env`; each env stepped for its whole
  segment before the next env starts. Buffer layout is per-env
  contiguous; GAE computed per segment. **Execution is sequential in the
  trainer process regardless of backend.**
- **SAC/DQN** (`sac_baseline.py:217`, `dqn_baseline.py:170`): round-robin
  — one step per env per iteration, still blocking per call.
- TCP mode DOES give process isolation (orchestrator spawns
  `main.py --headless --port P` via `HeadlessSimProcessPool`), but the
  trainer steps each connection synchronously — **no wall-clock
  parallelism anywhere in the codebase today**.
- Phase 5 measurement stands: sequential stepping does not scale
  (~394 → ~363 total sps at 4 envs).

## 4. TCP protocol (current reality)

- `sim_net/protocol.py`: NDJSON envelope `{type, payload}`,
  `PROTOCOL_VERSION = "2.0"`, `SUPPORTED_PROTOCOL_VERSIONS = ("2.0",)`.
- MessageType: HANDSHAKE, HANDSHAKE_ACK, DISCOVER_CONTRACT, CONTRACT_ACK,
  RESET, RESET_ACK, STEP, STEP_ACK, GET_STATE, STATE_ACK, ERROR.
- **No scenario/environment update command exists.** No CLOSE/SHUTDOWN.
- Server (`sim_net/server.py`): single client, if/elif dispatch,
  version negotiation picks highest mutual version. No auth (LAN trust).
- Headless mode is the same `SimulationStudioApp` class minus graphics;
  all stepping is RPC-driven (`server.poll_and_process()` per loop).
- **No `set_scenario` API on `SimulationEnvironment`.** Scenarios apply
  at construction (`build_env_from_dicts`) or via direct
  `env.scenario_def = ...` assignment in the UI (`app.py:182`).

## 5. Dataset format (current reality)

`transitions_v1` = `manifest.json` + `episodes.jsonl` (one episode/line).

- Episode record: `{episode_id, seed, env_fingerprint, scenario_id,
  total_return, length, termination_reason, steps: [...]}`.
- Step whitelist (hardcoded `dataset.py:141-148`): obs, action, reward,
  terminated, truncated, termination_reason. Diagnostics structurally
  excluded — VERIFIED.
- `schema_hash` = sha256({observation_schema, action_schema})[:16] taken
  from the trajectory header; first accepted episode sets it, later
  mismatches counted in `skipped.fingerprint_or_schema_mismatch`
  (**fingerprint and schema mismatches conflated in one counter**).
- `_read_episode` treats line 1 as header positionally — does NOT check
  `rec["type"]` (unlike `TrajectoryReader`, which does).
- `seed` in the trajectory header is the **run base seed**, not the
  per-(env,episode) reset seed — episode-level seeds are not
  reconstructable today.
- No dataset validator, no splitter, no statistics tooling exists.

## 6. Worker architecture (current reality)

- `scheduler.py`: fixed `workers` list at construction; `Worker` states
  include OFFLINE/FAILED/STOPPING **but nothing ever sets them**.
- Worker adapter duck type: `capabilities()`, `launch(...)`,
  `poll(exp_dir, run_id)`, `cancel(exp_dir, run_id)`.
- `remote_worker.py`: WorkerService (TCP, one request/response per
  connection, thread-per-conn, 64 MB cap) + RemoteWorkerAdapter.
- **No registration, no heartbeat, no leases, no reclaim, no worker
  registry, no duplicate-dispatch protection.** A dead remote worker
  leaves its job RUNNING forever.
- Worker needs experiment files on its own filesystem (`experiment_dir`
  path passed verbatim; root-confined on the worker's own
  `experiments_root`). No artifact transfer.
- Security present: mandatory token, constant-time compare,
  auth-before-dispatch, trainer module whitelist in
  `orchestrator._resolve_trainer`, argv-list subprocess (no shell),
  run-dir confinement check. Gaps: POLL/CANCEL `experiment_dir` not
  root-checked; token cleartext (documented LAN-only).
- `RetryPolicy.retryable_error_types` field stored but
  `is_retryable_error` consults the module-level frozenset — minor bug.

## 7. Trainer capabilities (current reality)

`TRAINER_CAPABILITIES`: ppo/sac = continuous actions, dqn = discrete;
all declare `multi_env: true`, `observation_types: ["vector"]`,
checkpoints, evaluation. No parallel-execution-mode declarations.

## 8. Curriculum (current reality)

`CurriculumController`: deterministic advancement from eval aggregates;
stage → scenario dict via `stage_scenario_dict()` +
`environment_overrides` mapping; `stage_seed(i) = base + i + stage*10000`.
On advancement trainers **rebuild envs** and call `runner.set_envs()`
(count must stay constant). `validate_contract` hard-rejects
`env_mode="tcp"` + curriculum (`trainer_contract.py:112-116`) — VERIFIED
limitation.

## 9. UI (current reality)

- TRAIN tab lives inside `inspector.py` (provider/action pattern;
  `train_provider` + `on_train_action` in `app.py`).
- Only chart: 138×18 px sparkline `PropertyRow("chart")`, single series,
  no axes, no multi-series. Comparison view is a text table.
- No workers view; dataset section shows export counts only.
- UI calls services only — clean layering confirmed.

## 10. Legacy paths (current reality)

Per `LEGACY_PATHS_ANALYSIS.md`: legacy env runtime, legacy
`ObservationSchema`/`spaces.py`, `*.sim.json` format, `sim_client`
baselines — all RETAINED with dependents. Note the doc references
`sim_client/train.py`; the actual file is
`sim_client/agents/ppo_train.py` (doc drift to fix).
`sim_env/scenarios.py` (legacy ScenarioConfig) still present.

## 11. Known limitations carried into Phase 6 (verified real)

1. No wall-clock env parallelism anywhere (sequential stepping both in
   PPO segments and SAC/DQN round-robin; TCP envs stepped one-at-a-time).
2. `env_mode="tcp"` + curriculum rejected (no scenario swap on wire).
3. No convergence/learning-quality evidence for any algorithm.
4. No imitation-learning consumer for transitions_v1.
5. Trajectory capture = first-N-episodes-per-env prefix only; no
   sampling/filters; no episode summary/footer record.
6. Workers: no registration/heartbeat/lease/reclaim; OFFLINE state dead.
7. Comparison charts not rendered (sparkline only); no worker/dataset UI.
8. DQN eval = greedy argmax only; replay buffer not checkpointed;
   batch scheduler in-process only.
9. `dataset._read_episode` ignores record `type`; fp/schema mismatch
   counters conflated; header seed = run seed not episode seed.
10. Legacy paths retained untouched; `RetryPolicy` field ignored.
11. `eval_env` replay capture uses `eval_env.vehicle` — inprocess-only
    (hasattr-guarded, so TCP eval just skips replays).
12. Benchmarks dir has phase4/5 scripts; no learning benchmarks.

## 12. Performance bottlenecks (baseline)

- Phase 5 measured: 1 env inprocess ≈ 394 sps; 4 envs sequential ≈
  363 total (~91/env) — CPU-bound single process (GIL + physics).
- PPO subprocess 512 steps: 232 sps.
- Process startup for headless TCP sims: ~1.6 s spawn (Phase 5).
- No process-pool backend exists to measure yet.

## 13. Environment facts for Phase 6 work

- Python 3.13, torch 2.10.0+cpu, pygame+ModernGL UI. Windows host —
  `multiprocessing` requires `spawn` context.
- `SIMULATOR_VERSION = "4.0.0"`; wire protocol "2.0"; worker protocol
  "1.0"; contract "1.0"; manifest "1.0"; dataset "transitions_v1".
- DQN-capable preset: `presets/custom_environment.sim.json` (discrete
  5-action space_type). Continuous presets: oval_circuit, serpentine,
  obstacle_challenge (all `agent: null` legacy — how do they get
  obs/action schemas? via legacy paths).
- Scenario library: `ScenarioDefinition.get_standard_scenarios()` —
  basic_lane_following, high_speed_racing, wet_adverse_weather,
  obstacle_evasion, sensor_noise_challenge, full_domain_randomization.
- Scenario switching does NOT require track rebuild (ScenarioDefinition
  is geometry-independent); `env.reset()` already re-reads scenario_def.
  **Bug risk**: `reset()` appends `obstacle_overrides` entities without
  clearing previous scenario entities (`environment.py:282-294`) —
  `set_scenario` must handle entity cleanup.

## Discrepancies vs Phase 5 report

- Report is accurate on all checked claims. Minor doc drift:
  `sim_client/train.py` in legacy doc = actually
  `sim_client/agents/ppo_train.py`. `benchmarks/` exists (not empty).
- `HeadlessEnvPool` exists in `headless.py:57` but is unused.

## Phase 6 starting position

Architecture is clean enough to extend without rewrites:

- Parallelism: add a VecEnv duck-type (`num_envs`, `reset_all`,
  `reset_at`, `step_all`, `set_scenario`, `close`) consumed by runners
  via additive code paths; `ProcessVectorEnv` backend via
  `multiprocessing` spawn + Pipe; new `env_mode="process"`.
- TCP curriculum: protocol 2.1 + SET_SCENARIO command +
  `SimulationEnvironment.set_scenario()` (with entity cleanup).
- Everything else layers onto existing contract/scheduler/dataset/analytics.
