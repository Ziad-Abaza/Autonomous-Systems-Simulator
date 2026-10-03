---
noteId: "d5628170bf6911f1a29f1fbaabbd87c8"
tags: []

---

# MASTER BLOCKERS

Severity: **Critical** prevents core simulator purpose/operation · **High** prevents
reliable research/training/use · **Medium** workaround exists · **Low** quality/debt.

## Critical

### B-C1 — Stationary-policy reward exploit → no demonstrated learning
- **Symptom:** every training artifact on disk + a fresh PPO run this audit produce a policy that sits still and collects reward.
- **Root cause:** default reward pays ~0.79/step for staying centered/aligned at rest — nearly equal to driving (0.83/step measured), with zero risk. `term_completion` also disabled in all templates.
- **Evidence:** `experiments/exp_38eba258e68bf064` (this audit): 2×1501-step episodes, mean_speed 0.008–0.013 m/s, return ≈0.79/step; `experiments/baseline_ppo/metrics.json` 54/54 timeouts; live idle-vs-drive probe (0.790 vs 0.831/step).
- **Affected:** PH2-RL-010, PH5-TRN-*, PH6-BEN-007, ARL-009 — the platform's stated purpose.
- **Consequence:** no proof any agent can learn to drive here.
- **Required fix:** complete agentRL (anti-exploit reward exists + verified; needs trained E001 run mean_speed>2 m/s, completions>0).

### B-C2 — Standalone packaging structurally broken
- **Symptom:** no dist/ build can currently produce a correct app.
- **Root cause:** no `sys._MEIPASS`/frozen handling → `repo_root` resolves into `_internal/`; presets bundled at wrong level; `torch` excluded while `sim_ui` lazily imports trainers; `dynamics_panel.py` imports unpackaged `tools/`; `datas=[]`.
- **Evidence:** `AI_Environment_Simulator.spec`, `build/package_windows.py`, `app.py` path code; packaging audit.
- **Affected:** PH0-PKG-018 (original headline requirement).
- **Consequence:** the platform cannot ship.
- **Required fix:** dedicated packaging phase (data-root resolution, torch policy, tools dep, rebuild+clean verify).

## High

| ID | Symptom | Root cause | Evidence | Affected | Consequence | Required fix |
|----|---------|------------|----------|----------|-------------|--------------|
| B-H1 | TCP clients get wrong spaces for authored agents | HANDSHAKE returns legacy `action_config`/`observation_schema` | `env_handler.py:68-70`, `gym_env.py:34-65` | PH0-NET-013, PH1-ENV-007 | silent mis-shaped training over TCP | serve effective contract or prefer DISCOVER_CONTRACT |
| B-H2 | Open tracks never finish; road extends virtually past end | `term_completion` off in all templates; projection clamps to end sample → lateral offset vs terminal tangent | template matrix: `empty`/`slalom` 1800-step "running"; `tracks/open_track_overshoot.txt`; `spline.py:203-239` | PH1-TRK-009, PH75-TPL-012 | unbounded episodes, wrong on-road semantics | enable completion on open templates; end-of-course boundary |
| B-H3 | step-after-done returns terminated AND truncated | `environment.py:372-381` both-flags return | live `invalid_call_after_done` observed | PH1-ENV-004 | breaks trainer bootstrap semantics (agentRL risk #3) | single-flag or documented error + tests |
| B-H4 | Headless mode never records TCP steps | `_record_step_if_needed` only in GUI branch | `app.py:1169` vs `1182-1184` | PH0-REC-015, PH7-REC-006 | `--headless` runs can't capture episodes | move hook into step path |
| B-H5 | Disabled sensor leaves orphaned obs channel | suite drops sensor; channel remains, fills constant `[1.0]*15`; validator catches ERROR but runtime tolerates | `_ev_lidar`/`_ev_val` evidence scripts | PH75-SEN-007 | agents can silently train on dead channels | fail-fast at env build or auto-prune channel |
| B-H6 | E001 experiment running; T14 matrix committed mid-audit | `experiments/matrix.py` landed (`99d3083`); test fixed (69/69); E001 in flight with early learning signal (collision-terminating episodes at ~2 m/s mean speed by 15k steps) | live `agentRL/experiments/runs/E001/` metrics | ARL-009 | gate not yet passed — watch for eval verdict | complete the matrix run + E002–E010 |

## Medium

| ID | Symptom | Root cause | Evidence | Affected | Fix |
|----|---------|------------|----------|----------|-----|
| B-M1 | `cli batch` creates runs never dispatched | expansion writes CREATED runs only | `cli.py` cmd_batch | PH4-BAT-010 | route to scheduler or remove |
| B-M2 | SimServerMulti recycles dirty envs | no auto-reset on client checkout | `multi_server.py:184-194` | PH6-VEC-006 | re-reset on checkout |
| B-M3 | Unbounded `_rx_buffer`; no server timeouts | no cap/timeout logic | `server.py:108-124` | PH0-NET-012 | cap buffer + idle timeout |
| B-M4 | Dead UI controls still mutate dirty-flag | scen_*/act_min/max rows lack handlers; sp_*/vc_* handlers lack emitters | `inspector.py` handlers vs emitters | PH3-DSN-003, PH3-UI-011, PH1-UI-010 | wire or remove both directions |
| B-M5 | Replay zeroes lateral velocity; no entities | `app.py:963` `vel_body=Vec2(speed,0)` | code | PH0-REC-015 | store vel_body in frames |
| B-M6 | Recording FIFO silent drop + missing termination_reason sidecar | 5000-cap `recorder.py:78-99`; `app.py:908` | code | PH7-REC-006 | truncation marker + sidecar field |
| B-M7 | Scenario dead fields + leaks | weather/light/ambient/ScenarioConfig.obstacles/global_seed unused; target_speed_override never restored | `environment.py:263-342`; `randomization_designer.py:143` | PH0-ENV-011, PH3-DSN-006 | consume/document/remove; restore target_speed |
| B-M8 | Entity-create exceptions swallowed | `try/except: pass` | `environment.py:341-342` | PH3-DSN-006 | log/surface |
| B-M9 | Theme violations | hud.py ~40 + editor.py ~14 hardcoded literals | screenshots on light theme | PH7-THM-002 | tokenize |
| B-M10 | Dual-path key/reason drift | lap/completion, backward/reverse, lap_completed/completion, 108°/100° | both engines | PH0-ENV-008/009 | unify names or document map |
| B-M11 | DATA tab per-frame scans; export unreachable | `os.walk` per frame (`datasets_panel.py:122`); export only in TRAIN | code | PH7-DAT-008 | throttle + export action |
| B-M12 | PPO residuals | act_dim=3 hardcoded; unclamped action stored; eval adapter hardcodes clips | `ppo_baseline.py`, `evaluation.py` | PH4-PPO-015 | contract-driven dims |
| B-M13 | `SpawnMode.RANDOM_CHECKPOINT` dead | enum vs `environment.py:305` | code | PH3-DSN-006 | implement or remove |
| B-M14 | `SimGymEnv.get_state` ignores msg_type | `client.py:105-108` | code | PH0-NET-013 | assert msg_type |

## Low

| ID | Item |
|----|------|
| B-L1 | Smoke tools exit 1 despite all-pass (reporting bug) |
| B-L2 | Flaky tests: `test_cancel_running_trainer` (known), `test_sac` nan_guard (observed once) |
| B-L3 | Keyboard nav absent for menus/dialogs; Enter accept only on new_track |
| B-L4 | ui_scale scales fonts only (fixed-px metrics) |
| B-L5 | Unbounded text: status_toast raw exceptions, inspector labels, replay chip |
| B-L6 | PiP overflow >3 cameras; SCENE list cap 8; OVERVIEW 3/7 templates |
| B-L7 | Dead code: legacy HUD fns, dead imports environment.py:21-27, orphan scripts experiments/, presets dual-source |
| B-L8 | Empty `supported_versions` silently assumes 2.1 (`env_handler.py:34-41`) |
| B-L9 | `info` dict carries diagnostic geometry to TCP clients (documented-intentional; document in client docs) |
| B-L10 | Terminology drift: DATA/Datasets/Trn; inspector abbreviations |
| B-L11 | Missing tests: replay player, dataset scan, recording dest, packaging |
| B-L12 | README/docs drift: test count says "390+" vs actual 492; phase6 perf numbers pre-VD-fix |

---

# REPAIR-PHASE ADDENDUM (2026-10-04)

Post-audit implementation pass. Historical entries above are preserved verbatim;
this section records disposition only.

| Blocker | Disposition | Evidence |
|---------|-------------|----------|
| B-C1 stationary reward | RESOLVED — motion-gated alignment components; idle −0.02/step, drive +0.42/step; learn-bench 0→5.2 m/s (+797 return) in 6k PPO steps | `tests/test_reward_antiexploit.py` (19), `benchmarks/phase6/benchmark_results.json` |
| B-C2 packaging | RESOLVED — `sim_project/paths.py` (_MEIPASS/runtime roots), spec datas+hiddenimports, frozen `--module` trainer passthrough, dist build 236.8 MB executed + TCP-verified | `dist/AI_Environment_Simulator/`, packaged handshake/step evidence |
| HANDSHAKE legacy spaces | RESOLVED — handshake serves authored agent contract | `tests/test_handshake_contract.py` |
| Open-track end/completion | RESOLVED — endpoint bounds + completion enabled on open templates | `tests/test_track_semantics.py`, template matrix |
| step-after-done both-flags | CONFIRMED-BY-DESIGN — documented sentinel `invalid_call_after_done`; was a misclassified defect | `tests/test_episode_semantics.py` |
| headless recording | RESOLVED — externally-driven steps recorded incl. vel_body; FIFO truncation now marked | `tests/test_headless_recording.py` |
| orphaned obs channels | RESOLVED — runtime refuses unattached sources (was silent constants) | sensor family tests (118) |
| cli batch inert | RESOLVED — delegates to canonical batch-run scheduler | `tests/test_cli_batch_alias.py` |
| remote workers unverified | RESOLVED — two-process TCP dispatch verified; found+fixed bind port 0 bug | `experiments/exp_6625d75692531f24` remote run |
| E001 learning gate | PROGRESS — E001c (60k) returns +207..+694 at 9–12 m/s sustained; collision-bound, not converged | `agentRL/experiments/runs/E001c/` |

## New defects discovered during repair
- `WorkerService` bound to port 0, silently ignoring `--port` (fixed — real two-process verification uncovered it).
- `local_roll`/`near_clip`/`far_clip` absent from the camera contract end-to-end (added).
- `global_seed` bypassed by injected rng (fixed — per-reset SeedSequence over [env_seed, global_seed]).
- `target_speed_override` leaked across scenario switches (fixed — baseline restore before override).
- Legacy `ScenarioConfig.obstacles` never spawned; entity errors swallowed (fixed — tagged spawn + `scenario_warnings`).
- Multi-server pooled envs recycled dirty (fixed — reset on checkout).
