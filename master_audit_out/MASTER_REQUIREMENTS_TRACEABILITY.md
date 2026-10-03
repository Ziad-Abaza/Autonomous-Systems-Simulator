---
noteId: "778e1eb0bf6911f1a29f1fbaabbd87c8"
tags: []

---

# MASTER REQUIREMENTS TRACEABILITY

Baseline: `docs/MASTER_PROJECT_PLAN_BASELINE.md`. Every requirement audited
requirement-by-requirement. Statuses per spec taxonomy. Evidence levels:
A runtime · B behavioral test · C source-verified · D doc-only · E historical · F none.
No COMPLETE is granted on D/E alone. `evidence/` paths under `docs/master_audit/evidence/`.

## Phase 0 — Original platform

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH0-CORE-001 | 0 | 60 Hz deterministic clock + seeded RNG | COMPLETE | `sim_core/clock.py:16-49`; `environment.py:384`; bitwise-equal 200-step trajectories (evidence: `_ev_sens.py` run) | A/B | `FixedClock` + `default_rng` | `test_phase1_hardening`, VD determinism test | identical-seed runs equal (this audit) | none | low |
| PH0-CORE-002 | 0 | Catmull-Rom spline + queries | COMPLETE | `spline.py:37-259`, `get_closest_point:203`, `sample_at_distance:245` | B | `TrackSpline` | spline/param tests | exercised in every env run | none | low |
| PH0-CORE-003 | 0 | Procedural mesh gen | COMPLETE | `mesh_generator.py` `TrackMeshGenerator`; SIMULATE screenshot (evidence/screenshots/s1) | A/B | mesh+boundaries+gates | mesh tests | renders in app | none | low |
| PH0-CORE-004 | 0 | Semantic↔geometry separation | COMPLETE | `road_definition.py` vs `mesh_generator.py`; editor edits road_def only | C | layered | — | — | none | low |
| PH0-VEH-005 | 0 | "4-wheel dynamic model w/ Pacejka" | COMPLETE | `tire_model="pacejka4"`: per-corner Fz (long+lateral transfer), wheel-slip w/ yaw offsets, magic-formula post-peak decay, per-wheel envelopes + relaxation + diagnostics; default `bicycle` kept as matrix baseline | A/B | both models live-validated | `test_vehicle_physics.py` (7, incl. 3 pacejka4) | `evidence/vehicle/pacejka4_60hz` + compare table | none — collision response stays axle-level by design | low |
| PH0-VEH-006 | 0 | OBB collision vs bounds/obstacles | COMPLETE | `collision.py` `VehicleCollisionChecker` + broadphase | B | SAT + spatial hash | broadphase + collision tests | collisions terminate episodes live | none | low |
| PH0-SEN-007 | 0 | Sensor suite independent rates | COMPLETE | `sensor_manager.py`, sensors 4 types; SENSORS tab screenshot | A/B | state/lidar/camera/imu | sensor tests | screenshot `e2_insp_sensors_two_cams` | procedural camera ignores pose/FOV (see PH75-SEN-009) | low |
| PH0-ENV-008 | 0 | Composable reward + decomposition | COMPLETE | `reward_engine.py`, `reward_designer.py:319-330`; `info['reward_breakdown']` live | A/B | both engines | reward tests | HUD screenshot shows decomposition | key-name drift legacy↔compiled | low |
| PH0-ENV-009 | 0 | Attributed termination | COMPLETE | `termination_engine.py`, `termination_designer.py:198-226` | B | reasons enum+rules | termination tests | reasons observed live (collision/off_road/max_duration) | reason-name drift between paths | low |
| PH0-ENV-010 | 0 | Seedable domain randomization | PARTIAL | `domain_randomizer.py`, `randomization_designer.py:143` — `global_seed` bypassed (env injects clock.rng) | B | mass/friction/noise/spawn jitter | randomization tests | — | honor or remove `global_seed` | low |
| PH0-ENV-011 | 0 | Scenarios (weather/light/obstacles) | PARTIAL | `environment.py:263-342` consumes friction/noise/time-limit/spawn/target-speed/obstacles; `weather/time_of_day/ambient_light` serialized but dead at runtime; legacy `ScenarioConfig.obstacles` never spawned | B | `scenario_designer` + env consumption | `test_scenario_control` | scenario swap tests pass | dead fields policy; entity-create errors swallowed; target_speed leaks across switches | medium |
| PH0-NET-012 | 0 | TCP NDJSON protocol | COMPLETE | `protocol.py:28`, `server.py`, `env_handler.py`; tcp_recording_e2e 15/15 re-run | A/B | NDJSON+handshake | `test_external_protocol`, `test_tcp_multi` | e2e harness | rx-buffer cap, idle timeout | low |
| PH0-NET-013 | 0 | SimGymEnv adapter | IMPLEMENTED-BUT-BROKEN | `gym_env.py:34-65` sizes spaces from HANDSHAKE, but `env_handler.py:68-70` returns **legacy** spaces when compiled agent active → obs/action shape mismatch for authored agents | B/C | adapter real | `test_gym_integration` (default space only) | — | HANDSHAKE must serve effective contract | high |
| PH0-NET-014 | 0 | Reference agents | COMPLETE | `agents/pid_driver.py`, `random_agent.py`, `ppo_train.py` (legacy rollout), `ppo_baseline.py` (real) | B | 3+ agents | `test_full_rl_loop` | PID drives in tests | ppo_train eager torch import (packaging risk) | low |
| PH0-REC-015 | 0 | Recording + replay | PARTIAL | `recorder.py`, `replay.py`; e2e 15/15; **but** headless path never records (`app.py:1169` vs `1182-1184`), replay zeroes vy (`app.py:963`), 5000-frame silent FIFO drop (`recorder.py:78-99`) | A/B | recorder+player | tcp e2e | replay loaded + pose applied live | headless hook, vy fidelity, truncation marker | medium |
| PH0-UI-016 | 0 | Interactive studio UI | COMPLETE | `sim_ui/*`; smoke 31/31; screenshots set | A | home/workspace/tabs | `test_ui_widgets` | app boots + drives (smoke) | none for core scope | low |
| PH0-PRJ-017 | 0 | .sim.json persistence | COMPLETE | `serializer.py` schema 1→3; roundtrip tests | B | serializer+migrations | schema tests | 17 files load live | presets dual-source | low |
| PH0-PKG-018 | 0 | Standalone zero-dep exe | IMPLEMENTED-BUT-BROKEN | spec + `package_windows.py` exist; **no dist/**; no `_MEIPASS` handling → data→`_internal/`; bundled presets at wrong level; torch excluded → UI training crashes; `tools/` not packaged (dynamics_panel); `datas=[]` | E(last verified P1 era)+C | build script | none | none — cannot verify | whole packaging phase | high |
| PH0-ED-019 | 0 | 2D spline editor | COMPLETE | `editor.py`, `editor_ui.py`; smoke-tested ops | A/B | select/draw/place | `test_edit_history` | drag/insert/close live | none | low |

## Phase 1 — Hardening

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH1-CTL-001 | 1 | Steering sign fix | COMPLETE | commit `74bdf40`; steering tests | B | — | steering tests | — | none | low |
| PH1-CTL-002 | 1 | PID indexing + stability | COMPLETE | `pid_driver.py` obs[8:23]; PID loop test | B | — | `test_full_rl_loop` | — | none | low |
| PH1-TIME-003 | 1 | Documented time model | COMPLETE | `docs/TIME_MODEL.md`; FixedClock fixed-step only (real-time path dead) | B | — | — | — | none | low |
| PH1-ENV-004 | 1 | Gym termination semantics + reasons | PARTIAL | `environment.py:372-381` returns terminated=True **AND** truncated=True on post-done step (live observed `invalid_call_after_done`) — violates single-flag contract | A/B | reason strings | phase1 tests | observed in audit run | single-flag or documented semantic | high for trainer correctness |
| PH1-ENV-005 | 1 | Action sanitization | COMPLETE | `spaces.py:50-91`, `action_designer.py:217-324`; NaN→0/clamp/deadzone/rate-limit | B | both validators | action tests | — | none (advisory-by-design) | low |
| PH1-ENV-006 | 1 | Schema governs all obs forms; no leakage | COMPLETE | compile-time `validate_no_leakage` (`observation_designer.py:331`) + validator re-check | B | gate | `test_critical_observation_leakage_rejection` | — | none | low |
| PH1-ENV-007 | 1 | SimGymEnv adapts vector/image/dict | IMPLEMENTED-BUT-BROKEN | same HANDSHAKE defect as PH0-NET-013 — adapter builds from stale legacy schema | C | adapter | — | — | effective-contract handshake | high |
| PH1-RWD-008 | 1 | Directional checkpoints + teleport clamp + reverse penalty | COMPLETE | `checkpoint.py` directionality; `delta_s>5→0` both engines | B | guards | reward tests | — | threshold drift 108°/100° | low |
| PH1-TRK-009 | 1 | Open-track course completion | IMPLEMENTED-BUT-BROKEN | mechanism exists (`termination_designer.py:200`) BUT (a) every shipped template has `term_completion enabled:false` (verified all 7); (b) past open end `get_closest_point` clamps to last sample → `is_on_road=True` while driving straight past finish (evidence: `tracks/open_track_overshoot.txt`, template matrix `empty`/`slalom` ran 1800 steps "running") | A | mechanism | open-track tests exist | live repro | enable completion in open templates; end-of-course boundary semantics | high |
| PH1-UI-010 | 1 | Inspector + spawn placement + save/load | PARTIAL | inspector 14 tabs work (screenshots); BUT `sp_*` spawn handlers exist with no emitting rows (dead) — spawn yaw/elev/speed uneditable anywhere; validator suggests non-existent "heading presets" | A/B | inspector | smoke | — | wire or remove spawn props | medium |

## Phase 2 — Authoring studio

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH2-ENT-001 | 2 | Generalized entity model | COMPLETE | `entity.py` `ENTITY_CLASS_MAP` + `create_entity` | B | 5+ entity types | entity tests | placed+dragged in smoke | none | low |
| PH2-ED-002 | 2 | Point insert + width/elev/bank/curvature gizmos | COMPLETE | edge insert + drag + POINT tab verified; per-point width/elevation/banking **drag handles** + entity rotation handle (was dead `is_dragging_width`/`is_rotating_entity` flags); curvature viz + tangent arrows + bank labels drawn | A/B/C | editor | `test_editor_gizmos.py` (6) | headless drag verified (12→32m, +5m, +5°, 90°) | none | low |
| PH2-ED-003 | 2 | Entity placement/rotation gizmos | COMPLETE | place tool + drag + entity tab edit | A/B | editor | — | smoke | none | low |
| PH2-ED-004 | 2 | Scene hierarchy bidirectional | PARTIAL | SCENE tab ↔ canvas selection works (`app.py:1043-1046`); **list hard-capped at 8 entities** (`inspector.py:522`) | A/C | scene tab | — | smoke | uncap or scroll list | low |
| PH2-RWD-005 | 2 | Reward config UI + validation | COMPLETE | REWARD tab weight rows + validator gating | A/B | inspector | — | screenshot `e2_insp_reward` | none | low |
| PH2-SEN-006 | 2 | Sensor suite manager UI + obs preview | COMPLETE | SENSORS tab (7.5): add/remove/dup/enable/params per type; obs contract updates live (smoke check) | A/B | inspector+`handle_sensor_prop` | phase75 tests + smoke | "SENSORS add camera" PASS | in-inspector preview (PiP is the preview) | low |
| PH2-COL-007 | 2 | Spatial hash broadphase | COMPLETE | `SpatialHashGrid2D` + `benchmarks/broadphase_results.json` | B | broadphase | benchmark tests | — | none | low |
| PH2-VIS-008 | 2 | Vision pipeline isolation + benchmark | COMPLETE | `benchmark_vision.py` + results; debug overlays excluded from FBO | B/C | offscreen pipeline | — | — | per-camera render gating | low |
| PH2-PRJ-009 | 2 | Schema 2.0 + migration | COMPLETE | serializer migrations; tests | B | — | migration tests | — | none | low |
| PH2-RL-010 | 2 | Real PPO + reproducible baseline artifacts | PARTIAL | `ppo_baseline.py` = real CleanRL-grade PPO (verified code); `experiments/baseline_ppo/metrics.json` = real 49k-step run — **but all artifacts show stationary policies** (54/54 timeouts, ~0 speed); this audit ran a fresh 4,096-step PPO run (evidence/experiments/exp_38eba258e68bf064) → 2×1501-step episodes, mean_speed 0.008–0.013 m/s, return≈0.79/step = idle reward | A | trainer+runner | ppo tests | **live PPO run executed** | demonstrated learning (→agentRL) | high — purpose unproven |
| PH2-QA-011 | 2 | Tests green + docs set | COMPLETE | 492 passing; docs exist | B | suite | suite | — | none | low |

## Phase 3 — RL environment designer

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH3-DSN-001 | 3 | AgentDefinition | COMPLETE | `agent.py:64-127`; `sensor_configs` serialized source-of-truth | B | dataclass | agent tests | — | none | low |
| PH3-DSN-002 | 3 | Observation designer + leakage gate | COMPLETE | `observation_designer.py`; compile gate + validator | B | designer+pipeline | obs tests incl. leakage | — | none | low |
| PH3-DSN-003 | 3 | Action designer + compiled decoder | PARTIAL | `action_designer.py` real; **but** inspector `act_min/act_max` stepper rows have no handlers (dead controls) | B/C | designer | action tests | — | wire rows | low |
| PH3-DSN-004 | 3 | Reward designer + guards | COMPLETE | `reward_designer.py` incl. teleport/reverse | B | designer | reward tests | — | none | low |
| PH3-DSN-005 | 3 | Termination designer + trunc/term split | COMPLETE | `termination_designer.py` rules+flags | B | designer | term tests | — | none | low |
| PH3-DSN-006 | 3 | Scenario + randomization designers | PARTIAL | designers real; dead fields (weather/light/ambient, ScenarioConfig.obstacles, global_seed); entity-create exceptions swallowed (`environment.py:341-342`); `target_speed_override` never restored | B/C | designers | scenario tests | — | consume/document/remove dead fields; surface errors; restore target_speed | medium |
| PH3-DSN-007 | 3 | Curriculum model | COMPLETE | `curriculum.py` declarative (runtime in P5) | B | model | curriculum tests | — | none | low |
| PH3-VAL-008 | 3 | Validator + readiness gate | COMPLETE | `validator.py`; ERROR correctly fired for orphan obs channel (this audit: `lidar_ranges` → ERROR) + spawn-on-road | A/B | validator | validator tests | live ERROR observed | nav-to-field; SCENE subsystem mapping | low |
| PH3-VAL-009 | 3 | Versioning/fingerprint/diff | PARTIAL | versioning+fingerprint real; diff still compares legacy keys (P3 audit §5.2.9 — unverified whether fixed) | B/C | versioning | versioning tests | — | verify diff covers Phase-3 fields | low |
| PH3-VAL-010 | 3 | Templates + training export | COMPLETE | 7 templates (validated); `export.py` bundle | A/B | templates+export | template tests | template matrix run | none | low |
| PH3-UI-011 | 3 | Designer inspector tabs | PARTIAL | all tabs present + most functional; dead controls: scen_select/weather/time rows, act_min/max (clickable, no handler, still dirty doc) | A/B | inspector | — | screenshots per tab | wire or remove dead rows | medium |
| PH3-RT-012 | 3 | Compiled pipelines at runtime | COMPLETE | `environment.py:104-165` compiled iff agent set | B | pipelines | env tests | live obs (23,) | none | low |
| PH3-PRT-013 | 3 | DISCOVER_CONTRACT | COMPLETE | `env_handler.py:78-123` | B | handler | `test_protocol_discovery` | — | none | low |
| PH3-SER-014 | 3 | Schema 3.0 serialization | COMPLETE | serializer + agent to/from_dict | B | — | schema tests | — | none | low |

## Phase 4 — Experiment platform

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH4-EXP-001 | 4 | ExperimentManifest | COMPLETE | `manifest.py` + fingerprints | B | manifest | experiment tests | created exp_38eba258 live | none | low |
| PH4-EXP-002 | 4 | ExperimentManager | COMPLETE | `manager.py`; `manifest_version` gating (`:146`) — phantom filtered | A/B | manager | manager tests | `cli list` filters legacy | legacy dirs unmigrated | low |
| PH4-RUN-003 | 4 | Run lifecycle | COMPLETE | `run.py` states+transitions | B | run manager | run tests | run.json produced live | none | low |
| PH4-RUN-004 | 4 | Orchestrator launch/poll/cancel/timeout | COMPLETE | `orchestrator.py` subprocess, whitelist, terminate→kill, timeout→INTERRUPTED | A/B | orchestrator | orchestrator tests | dummy+PPO runs launched+completed live | in-memory proc tracking orphans on restart | medium |
| PH4-CON-005 | 4 | Trainer contract v1.0 | COMPLETE | `trainer_contract.py` run-relative + resolve | B | contract | contract tests | contract.json produced | — | low |
| PH4-MET-006 | 4 | Metrics JSONL | COMPLETE | `metrics.py` scoped writer; metrics.jsonl written live | A/B | metrics | metrics tests | real metrics.jsonl read | — | low |
| PH4-ART-007 | 4 | ArtifactRegistry | COMPLETE | `artifacts.py`; registry.jsonl in run | A/B | registry | artifact tests | registry file produced | — | low |
| PH4-TRJ-008 | 4 | Trajectories agent/diag split | COMPLETE | `trajectory.py` writer/reader | B | trajectory | trajectory tests | — | — | low |
| PH4-EVL-009 | 4 | EvaluationRunner | COMPLETE | `evaluation.py` frozen-policy + adapters for 4 algos | B | eval | eval tests | — | PPO eval adapter hardcodes clips | low |
| PH4-BAT-010 | 4 | Batch expansion | PARTIAL | `batch.py` expansion real; `cli batch` creates runs that never dispatch (use `batch-run`) | B/C | expansion | batch tests | — | route `batch` to scheduler or remove | medium |
| PH4-REP-011 | 4 | ReproducibilityChecker | COMPLETE | `reproduce.py` 12 checks | B | checker | reproduce tests | — | — | low |
| PH4-HDL-012 | 4 | Headless factory + pools | COMPLETE | `headless.py` build_env_from_dicts + pools | B | headless | headless tests | used in every audit script | — | low |
| PH4-CLI-013 | 4 | CLI | PARTIAL | ~28 subcommands real; `batch` misleading; worker-status/list thin | A/B/C | cli | cli tests | validate-env/create/launch/runs executed | fix or remove `batch` | low |
| PH4-UI-014 | 4 | TRAIN tab | COMPLETE | tab + providers (`inspector.py`); provider result ~1s-cached + `list_experiments`/`list_runs` TTL caches w/ write-invalidation — no per-frame FS reads; dataset export surfaced in DATA too | C | TRAIN tab | — | screenshot `e2_insp_train` | none | low |
| PH4-PPO-015 | 4 | External PPO trainer subprocess | COMPLETE | `ppo_trainer.py` + `ppo_baseline.py`; real run executed this audit (contract→metrics→ckpts→run_result) | A | trainer | ppo tests | exp_38eba258 COMPLETED | act_dim=3; unclamped stored action | low |
| PH4-HRD-016 | 4 | Phase-3 hardening (overrides/fingerprint/negotiation/metadata) | COMPLETE | environment.py consumption; `PROTOCOL_VERSION 2.1` + SUPPORTED; recorder metadata fields | B | — | scenario/protocol tests | — | — | low |

## Phase 5 — Scale-out

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH5-CUR-001 | 5 | Curriculum runtime (eval-driven stage advance) | COMPLETE | `curriculum_runtime.py` `CurriculumController` + trainer hooks all 3 modes | B | controller | `test_curriculum_training` | — | a real curriculum run artifact | medium |
| PH5-CUR-002 | 5 | Curriculum persistence + fingerprint guard | COMPLETE | `curriculum_runtime.py:255-284` + harness guard | B | persistence | — | — | — | low |
| PH5-TRN-003 | 5 | Capability declarations + compat check | COMPLETE | `capabilities.py` `check_compatibility` in launch | B | capabilities | capability tests | — | — | low |
| PH5-TRN-004 | 5 | Shared trainer harness | COMPLETE | `trainers/_harness.py` (~491 ln) used by all 4 | B | harness | harness tests | PPO run used it live | — | low |
| PH5-TRN-005 | 5 | SAC trainer | COMPLETE (impl) | `sac_baseline.py` real twin-Q/auto-α/replay | B | SAC | sac tests | — | any real training artifact | medium |
| PH5-TRN-006 | 5 | DQN trainer | COMPLETE (impl) | `dqn_baseline.py` real DQN (vanilla argmax target despite "Double" docstring) | B | DQN | dqn tests | — | any real training artifact | medium |
| PH5-BAT-007 | 5 | BatchScheduler | COMPLETE | `scheduler.py` dispatch/retry/reclaim/persist | B | scheduler | scheduler tests | — | no batches/ artifacts → unexercised | medium |
| PH5-WRK-008 | 5 | Local worker pool | COMPLETE | orchestrator subprocess workers | B | — | — | — | — | low |
| PH5-WRK-009 | 5 | Remote workers (LAN, auth, capability) | IMPLEMENTED-BUT-UNVERIFIED | `remote_worker.py` real TCP WorkerService + token auth + RemoteWorkerAdapter; **no remote run artifacts exist** | C | worker service+adapter | worker tests (handshake-level) | none | a real remote run | medium |
| PH5-ANA-010 | 5 | Comparison/analytics + multi-chart | COMPLETE | `inspector.py:1077` chart + `cli compare` | B/C | analytics | — | — | theme-compliant colors | low |
| PH5-DAT-011 | 5 | Trajectory→dataset export | COMPLETE (tools) | `dataset.py` transitions_v1 export/validate/split/stats | B | dataset | dataset tests | — | artifacts on disk; UI reach | medium |
| PH5-HRD-012 | 5 | ep_len/seed fixes | COMPLETE | `ppo_baseline.py:245-250` seeded resets; env-authoritative step | B | — | — | — | — | low |
| PH5-LEG-013 | 5 | Legacy migration analysis | COMPLETE | `docs/LEGACY_*.md` | E | docs | — | — | — | low |

## Phase 6 — Scalable/verifiable training

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH6-VEC-001 | 6 | VectorEnv + SyncVectorEnv | COMPLETE | `vec_env.py:59-111` | B | vec impl | `test_vec_env` | — | — | low |
| PH6-VEC-002 | 6 | ProcessVectorEnv (spawn, crash-isolated) | COMPLETE | `process_env.py:128` spawn ctx; EnvWorkerCrash; kill test | B | proc env | `test_process_env` (real proc kill) | — | — | low |
| PH6-VEC-003 | 6 | Runner vec integration + env_mode=process | COMPLETE | harness + contract wiring | B | — | `test_process_training` | — | — | low |
| PH6-SCN-004 | 6 | set_scenario + entity cleanup | COMPLETE | `environment.py` set_scenario + `_scenario_spawned` | B | — | `test_scenario_control` | — | — | low |
| PH6-PRT-005 | 6 | Protocol 2.1 SET_SCENARIO | COMPLETE | `env_handler.py:177-241`; negotiation; mid-episode reject | B | — | scenario-control tests | — | — | low |
| PH6-VEC-006 | 6 | SimServerMulti + tcp_multi | PARTIAL | `multi_server.py` real; pooled envs recycled **dirty** (no auto-reset, `:184-194`) | B | — | `test_tcp_multi` | — | re-reset on checkout | medium |
| PH6-BEN-007 | 6 | Learning benchmark + convergence verdicts | PARTIAL | `benchmark_runner.py` + `convergence.py` real; **`benchmarks/phase6/results/` absent — never executed** | C | runner+verdicts | `test_convergence` | none | execute learn-bench | medium |
| PH6-DAT-008 | 6 | Dataset tools | COMPLETE | `dataset.py` validate/split/stats + sampler + provenance | B | — | `test_dataset_tools` | — | — | low |
| PH6-BC-009 | 6 | BC over transitions_v1 | COMPLETE | `sim_experiment/bc/` + `bc_trainer`; true E2E test (PPO→dataset→BC→ckpt) | B | bc pkg | `test_bc` E2E | — | a BC run artifact | low |
| PH6-WRK-010 | 6 | Worker hardening | COMPLETE | `worker_registry.py` + heartbeat/lease/reclaim/root-check | B | registry | worker tests | — | — | low |
| PH6-UI-011 | 6 | WORKERS/dataset/chart UI | COMPLETE | workers rows (status/job/hb) + reward chart + comparison multichart + dataset preview render; actions added: `trn_w_serve` launches+registers a local worker, `trn_w_off_{i}` marks offline via WorkerRegistry | C | UI rows + actions | — | worker lifecycle actions wired | none | low |
| PH6-QA-012 | 6 | perf measured | COMPLETE | `PHASE_6_PERFORMANCE_RAW.json`: process 1056.9 sps @4 envs (stale but honest); this audit: raw env.step ~2281 sps, PPO train ~250 sps, SAC ~68 sps | A | — | — | this audit's measurements | refresh numbers post-VD-fix | low |

## Phase 7 — Studio UX

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH7-WDG-001 | 7 | Widget layer | COMPLETE | `widgets.py` UIContext/hit/z/modal; real tests | B | widgets | `test_ui_widgets` | smoke | — | low |
| PH7-THM-002 | 7 | Theme tokens dark+light | PARTIAL | `theme.py` 5 palettes+set_theme; **hud.py ~40 + editor.py ~14 hardcoded literals** → SIMULATE/EDIT stay dark on light themes (visible in screenshots) | A/B | theme | palette tests | light-theme screenshots | theme-complete HUD/editor | medium |
| PH7-LIB-003 | 7 | TrackLibrary + thumbnails | COMPLETE | `library.py` + `thumbnails.py` | B | library | `test_track_library` | home screenshots | — | low |
| PH7-SHL-004 | 7 | Home + workspace tabs | COMPLETE | `home_screen.py`, `workspace_screen.py` TABS=5 | A | shell | — | smoke+screenshots | — | low |
| PH7-EDT-005 | 7 | Editor canvas (fit/zoom/snap/undo) | COMPLETE | editor + editor_ui | A/B | editor | `test_edit_history` | smoke | — | low |
| PH7-REC-006 | 7 | Recording dialog + dest + TCP-step capture | COMPLETE | dialog+browse+manifest verified; dest now **persists** (`settings.recordings_dir` survives restarts); TCP-step capture verified headless (CP-05) | A/B | rec flow | — | e2e + screenshots | none | low |
| PH7-RPL-007 | 7 | Replay picker + transport | COMPLETE | workspace replay + applied pose verified | A | replay UI | — | e2e PASS `replay applied pose` | vy fidelity | low |
| PH7-DAT-008 | 7 | DATA tab | COMPLETE | `datasets_panel.py` works; scan TTL-cached 3s (`scan_datasets_cached`); "Export…" action on dataset detail (copytree to chosen dir) | A/C | panel | — | screenshot `d1_data_datasets` | none | low |
| PH7-DLG-009 | 7 | Dialogs + dirty guards | COMPLETE | dialogs.py; confirm/new/rename/record | A/B | dialogs | — | smoke+screenshots | Enter-accept only new_track | low |
| PH7-QA-010 | 7 | Smoke + shots harness | PARTIAL | both harnesses PASS (31/31, 15/15) **but exit code 1** (reporting bug) | A | tools | — | re-run this audit | exit-code fix | low |

## Phase 7.5 — Functional repair

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| PH75-TRK-001 | 7.5 | Closure gesture + toggle | COMPLETE | `editor.py:275-284`, `editor_ui.py:172-179`; smoke "draw click on first point closes track" PASS | A/B | — | phase75 tests | smoke | — | low |
| PH75-TRK-002 | 7.5 | FINISH marker (open tracks) | COMPLETE | `editor.py:524-534`; screenshot `e5_open_route_finish` | A/B | — | — | screenshot | not authorable (documented) | low |
| PH75-TRK-003 | 7.5 | Spawn on-road validation | COMPLETE | `validator.py:138-163` | B | — | phase75 tests | — | — | low |
| PH75-ENT-004 | 7.5 | ent_* handlers | COMPLETE | `inspector.py:806-823`; smoke "inspector entity edit mutates position" | A/B | — | phase75 tests | smoke | — | low |
| PH75-ENT-005 | 7.5 | Template entity names | COMPLETE | `templates.py` pos= kwarg; name test | B | — | — | — | — | low |
| PH75-SEN-006 | 7.5 | SensorConfig serialization | COMPLETE | `sensor_config.py`, `agent.py:64-127` | B | — | phase75 tests | configs roundtrip | — | low |
| PH75-SEN-007 | 7.5 | Runtime suite from configs | PARTIAL | `build_from_configs` works; **but** disabling a sensor leaves its obs channel orphaned → validator ERROR (correct) yet runtime fills constant `[1.0]*15` instead of refusing (live evidence `_ev_lidar`/`_ev_val`) | A/B | — | phase75 tests | live: dead channel observed | fail-fast or auto-prune orphaned channel | medium |
| PH75-SEN-008 | 7.5 | SENSORS inspector tab | COMPLETE | `inspector.py:241-324` full per-type params | A/B | — | phase75 + smoke | screenshot `e2_insp_sensors` | — | low |
| PH75-SEN-009 | 7.5 | Multi-camera | COMPLETE | `_wire_camera_sensors` per-cam FBO; `image_<name>` channels; PiP loop; smoke "camera in observation contract" + runtime rebuild | A/B | — | phase75 + smoke | screenshot two cams | near/far/roll; fallback pose; PiP>3 overflow | low |
| PH75-EXP-010 | 7.5 | Experiment path fixes | COMPLETE | `manager.py:146` gating (phantom filtered — verified `cli list`), relative contracts, `default_experiments_root`, toasts | A/B | — | phase75 tests | list filters legacy | legacy dirs remain on disk | low |
| PH75-TXT-011 | 7.5 | text_input caret fix | COMPLETE | `widgets.py:430-442` | B | — | widget tests | — | — | low |
| PH75-TPL-012 | 7.5 | New templates | PARTIAL | 7 templates exist + validate; **but all ship `term_completion enabled:false`** (verified) → open routes can never "finish"; `empty`/`slalom` never terminate in 1800-step drive (template matrix evidence) | A | templates | template tests | template matrix run | enable completion on open templates | high |

## Vehicle dynamics

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| VD-001 | VD | Tire stiffness correction | COMPLETE | `vehicle_config.py` Cf=80k/Cr=85k (~13-15/rad); audit rerun const-steer 0.10 = 3.77° sideslip | A | config | VD tests | `evidence/vehicle/audit_run` | — | low |
| VD-002 | VD | Friction ellipse | COMPLETE | `vehicle_model.py` u-clamp + sqrt(1-u²) cap; asserted in tests | A/B | model | ellipse asserts | — | — | low |
| VD-003 | VD | Longitudinal load transfer | COMPLETE | `cog_height` now consumed via m·ax·h/L | B | model | VD tests | — | — | low |
| VD-004 | VD | Signed slip denominator | COMPLETE | signed deadzone ±0.5 m/s | B | model | VD tests | — | — | low |
| VD-005 | VD | 240 Hz substeps | COMPLETE | `physics_substeps=4`; timestep-independence <1.5 m/4 s reported | B | model | VD tests | — | — | low |
| VD-006 | VD | Bounded low-speed regularization | COMPLETE | tau 0.05 s relaxation (was uncapped forcing) | B | model | VD tests | — | — | low |
| VD-007 | VD | Proper-acceleration IMU | COMPLETE | accel_body = specific force | B | model+imu | VD tests | — | — | low |
| VD-008 | VD | Config params exposed | COMPLETE | wheel_radius/yaw_inertia/splits/substeps in config | B | config | — | — | — | low |
| VD-009 | VD | Per-step tire diagnostics | COMPLETE | alpha/fy/fx/fz per axle on state | B | model | VD tests | — | — | low |
| VD-010 | VD | Maneuver harness + metrics | COMPLETE | `tools/vehicle_dynamics/`; this audit executed full suite fresh | A | harness | 13 VD tests | evidence/vehicle/audit_run | >30 m/s, mu<0.6 unvalidated; no post-peak; detection-only collision | medium |

## agentRL (in-flight)

| ID | Phase | Requirement | Status | Evidence | Lvl | Implementation | Tests | Runtime Verification | Missing | Risk |
|----|-------|-------------|--------|----------|-----|----------------|-------|----------------------|---------|------|
| ARL-001 | ARL | T1–T4 foundations | COMPLETE | `core/*`, `obs/*`, `act/adapter.py`, `rewards/presets.py`; anti-exploit math verified vs real `CompiledRewardEngine` | B | all 4 pkgs | test_core/obs_spec/action_adapter/reward_presets | — | — | low |
| ARL-002 | ARL | T5–T6 EnvFactory + ScenarioMutator | COMPLETE | `envs/factory.py`, `track_registry.py`, `scenario_gen.py`, `track_gen.py` | B | envs | test_env_factory, test_scenario_gen | — | — | low |
| ARL-003 | ARL | T7 rehearsal buffers | COMPLETE | `memory/replay.py` committed (c680176) | B | memory | test_replay | — | — | low |
| ARL-004 | ARL | T8–T9 SAC + PPO agents | COMPLETE | `algos/sac.py`, `algos/ppo.py`, `core/agent.py` (commits 4352025, 7b0230f) | A/B | agents | test_sac, test_ppo | **real SAC run executed** (evidence/agent/sac_smoke: 1500 steps, ckpts, real losses) | — | low |
| ARL-005 | ARL | T10 Trainer + metrics + checkpoints | COMPLETE | `train/trainer.py`, `train/metrics.py`, `checkpoints/io.py`; real run produced metrics.jsonl + latest/step ckpts | A | trainer | test_trainer | live run | resume-equivalence deep check | medium |
| ARL-006 | ARL | T11 eval suite + failure classifier + matrix | COMPLETE | `eval/{evaluate,failures,matrix}.py` (509abb1) | B | eval | test_eval | — | — | low |
| ARL-007 | ARL | T12 ContinualTrainer + retention report | IMPLEMENTED-BUT-UNVERIFIED | `train/continual.py` committed (c9a097e): phased training, all-track eval grid, retention/forgetting/transfer, resume | B/C | continual | test_continual | none — no continual run artifact | a real A→B→C run with retention numbers | high |
| ARL-008 | ARL | T13 lab scripts | COMPLETE | `tests/agent/{run_policy,evaluate_policy,compare_runs,inspect_episode,test_env_tcp}.py` (f177909) | B/C | scripts | — | — | — | low |
| ARL-009 | ARL | T14–T15 experiment matrix + E001 learning proof | PARTIAL | E001c DONE (60k, +904 best, 9.6 m/s); E003 DONE (+215 @15.6 m/s); E005 DONE (150k continual, retention+holdout evals); E006–E008 evals DONE (gen_loop_4: 5/5 clean 1500-step eps @12 m/s); E010 DONE (resume integrity); E002/E004/E009 in flight — E004 required a trainer fix (stale-env step-after-done, 79.6k failure events on first attempt). "Converged safe driving" is partially demonstrated: clean full-horizon driving verified on a holdout, collisions remain on standard tracks at speed | A | matrix + runs | 70/70 tests incl. `test_mixed_trainer_steps_current_env` | runs/E001c–E010 artifacts | E002/E004/E009 completion | medium |

## SUPERSEDED requirements

| ID | Original | Superseded by | Status | Evidence |
|----|----------|---------------|--------|----------|
| (goal-point model) | separate finish object | implicit last-checkpoint finish + FINISH badge | SUPERSEDED | PHASE_7_5 report "Track architecture decision" |
| `cli batch` dispatch | batch expansion only | `batch-run` + BatchScheduler | SUPERSEDED | cli.py (batch remains misleading — see PH4-BAT-010) |
| ppo_train.py | rollout script | ppo_baseline.py + contract trainers | SUPERSEDED | kept for back-compat; eager torch import is a wart |
| protocol "2.0" fixed | fixed version | negotiated 2.0/2.1 | SUPERSEDED | protocol.py:28-29 |
| legacy ExperimentConfig dirs | legacy schema | manifest_version-gated manager | SUPERSEDED | manager.py:146 |

## Baseline corrections noted

See `docs/master_audit/BASELINE_CORRECTIONS.md` — includes: PH0-VEH-005 "4-wheel" vs delivered
single-track model; PH1-TRK-009 mechanism-exists-but-disabled-in-templates; PH0-PKG-018
verified-only-in-Phase-1; ARL ledger staleness vs actual commits.

---

# REPAIR-PHASE STATUS CHANGES (2026-10-04)

Authoritative delta list — see `docs/REPAIR_IMPLEMENTATION_REPORT.md` §13 for
full REQ-ID|BEFORE|AFTER|ROOT CAUSE|FIX|TEST|RUNTIME EVIDENCE rows.

-> COMPLETE (25): PH0-NET-013, PH1-ENV-007, PH1-TRK-009, PH0-PKG-018,
PH5-WRK-009, ARL-007, PH1-ENV-004, PH0-ENV-010, PH0-ENV-011, PH0-REC-015,
PH1-UI-010, PH2-ED-004, PH2-RL-010, PH3-DSN-003, PH3-DSN-006, PH3-VAL-009,
PH3-UI-011, PH4-BAT-010, PH4-CLI-013, PH6-VEC-006, PH6-BEN-007, PH7-THM-002,
PH7-QA-010, PH75-SEN-007, PH75-TPL-012

REMAINING PARTIAL: ARL-009 only — E002/E004/E009 runs in flight at last
update; see `agentRL/AGENT_RL_FINAL_REPORT.md` §3 for measured status.

Resolved in the completion push (2026-10-04): PH0-VEH-005 (4-wheel
Pacejka model, suite-validated), PH2-ED-002 (CP width/elev/bank drag
handles + entity rotation handle, 6 tests), PH4-UI-014 (provider +
manager TTL caches), PH6-UI-011 (worker serve/mark-offline actions),
PH7-REC-006 (recordings dir persisted), PH7-DAT-008 (cached scans +
export action). Also fixed: MixedTrackTrainer stepped a stale env after
track switches (E004's first run logged 79,593 step-after-done
failures); regression test `test_mixed_trainer_steps_current_env` added.

Post-push counts: COMPLETE 134 / PARTIAL 1 (ARL-009, runs completing)
/ BROKEN 0 / UNVERIFIED 0 / SUPERSEDED 5 (of 136 active + 5 superseded).
Prior line preserved: "REMAINING PARTIAL (7)" — superseded by the above.
