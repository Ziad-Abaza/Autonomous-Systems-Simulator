---
noteId: "24d25a00bf6a11f1a29f1fbaabbd87c8"
tags: []

---

# MASTER COMPLETION PLAN

Executable roadmap built strictly from verified gaps (MASTER_GAP_ANALYSIS).
Ordered by dependency, not convenience. Task IDs CP-xx.

```
CP-01..05  Contract hardening        (prerequisite seam)
  └─ CP-06 agentRL T14–T15 + E001    (the learning proof)  ← recommended NEXT
       └─ CP-07..09 Training-evidence artifacts
            └─ CP-10..14 Residual gaps (recording/sensors/UI/scenario)
                 └─ CP-15 Packaging
                      └─ CP-16 Theme/accessibility polish
```

Rationale: contract semantics (post-done, HANDSHAKE, open-end) define what trainers
and evaluators measure — fix them before producing learning evidence that must be
trusted. agentRL is already the in-flight critical path. Packaging waits because a
broken-learning simulator isn't worth shipping yet.

---

## CP-01 — HANDSHAKE effective-contract fix

- **Requirements:** PH0-NET-013, PH1-ENV-007 (B-H1)
- **Problem:** TCP clients get legacy spaces for authored agents.
- **Root cause:** `env_handler.py:68-70` returns `env.action_config`/`env.observation_schema` regardless of compiled agent.
- **Implementation:** HANDSHAKE branches on compiled agent → report compiled obs/action contract; SimGymEnv prefers DISCOVER_CONTRACT when present.
- **Deps:** none. **Files:** `sim_net/env_handler.py`, `sim_client/gym_env.py`, `sim_net/protocol.py`.
- **Tests:** authored-agent TCP handshake space match; gym_env dict-space build.
- **Runtime:** client connects to authored env → spaces match compiled pipeline.
- **Acceptance:** regression test green; no existing test regressions.

## CP-02 — step-after-done single-flag semantics

- **Requirements:** PH1-ENV-004 (B-H3)
- **Problem:** post-done step returns terminated=True AND truncated=True.
- **Root cause:** `environment.py:372-381`.
- **Implementation:** return `terminated=True` only + `invalid_call_after_done` reason (documented choice), or raise — pick and document in `docs/ACTION_SCHEMA.md`.
- **Deps:** none. **Files:** `sim_env/environment.py`, docs, `sim_experiment/trainers/_harness.py` (bootstrap check).
- **Tests:** contract test; truncation-bootstrap unit test.
- **Acceptance:** documented single-flag result; trainers handle it.

## CP-03 — Open-track end semantics + template completion

- **Requirements:** PH1-TRK-009, PH75-TPL-012 (B-H2)
- **Problem:** road "extends" past open end; completion rule off in every template.
- **Root cause:** `get_closest_point` clamp + `is_on_road` from projected sample; template rule config.
- **Implementation:** (a) `past_end` flag when projected s==end & pos beyond endpoint → off_road or new reason; (b) `term_completion enabled:true` for open-route templates/presets; (c) validator WARNING for open track w/o completion rule.
- **Deps:** none. **Files:** `sim_core/track/track_queries.py`, `sim_env/termination_designer.py`, `sim_env/templates.py`, `presets/*.sim.json`, `sim_env/validator.py`.
- **Tests:** overshoot regression; template validation suite.
- **Runtime:** open template finishes at last gate.
- **Acceptance:** `empty`/`slalom` terminate on finish; regression green.

## CP-04 — Sensor-disable fail-fast / auto-prune

- **Requirements:** PH75-SEN-007 (B-H5)
- **Problem:** orphaned obs channel fills constants.
- **Implementation:** env build refuses when enabled channel's sensor missing (or auto-prune + explicit warning); keep validator ERROR.
- **Files:** `sim_env/environment.py` suite-build path, `observation_designer.py`.
- **Tests:** disabled-lidar build raises/prunes.
- **Acceptance:** no silent constant channels.

## CP-05 — Contract hygiene batch

- **Requirements:** PH6-VEC-006, PH0-NET-012, PH4-CLI-013, PH4-PPO-015, PH0-ENV-011 (B-M1..M5, M14)
- **Items:** SimServerMulti re-reset on checkout; `_rx_buffer` cap + idle timeout; `cli batch`→scheduler-or-remove; PPO act_dim from contract + clamped action storage; scenario dead-field policy (document vs consume); `get_state` msg_type assert; entity-create error surfacing; `target_speed` restore on set_scenario; RANDOM_CHECKPOINT implement-or-remove.
- **Deps:** none. **Files:** `multi_server.py`, `server.py`, `cli.py`, `ppo_baseline.py`, `environment.py`, `episode_config.py`, `client.py`.
- **Tests:** per-item. **Acceptance:** batch green.

## CP-06 — agentRL T14–T15: experiment matrix + E001 ← NEXT

- **Requirements:** ARL-007 (partially), ARL-009 (B-H6, B-C1)
- **Problem:** `experiments/matrix.py` schema vs test (`KeyError 'algo'`); no E001 run.
- **Implementation:** finish matrix (align config keys), run E001 (≥60k steps, anti-exploit reward), produce `experiments/E001_*` artifacts + convergence verdict + `EXPERIMENTS.md`; run ContinualTrainer A→B→C → `continual_report.json` with retention/forgetting numbers.
- **Deps:** CP-02/03 strongly advised (training verdict must measure correct semantics). **Files:** `agentRL/experiments/`, `agentRL/train/continual.py` usage, `agentRL/runs/` artifacts.
- **Tests:** `test_experiment_matrix` green; suite ≥ all-green.
- **Runtime validation:** `run_policy.py` on a trained ckpt shows movement; `evaluate_policy.py` reports mean_speed>2 m/s, completions>0.
- **Acceptance:** E001 gate met with artifacts + report; retention measurable.

## CP-07 — Headless recording hook + replay fidelity

- **Requirements:** PH0-REC-015, PH7-REC-006 (B-H4, B-M5, B-M6)
- **Implementation:** `_record_step_if_needed` into step path; store vel_body; truncation marker; termination_reason sidecar; persist record dir.
- **Deps:** none. **Files:** `app.py`, `recorder.py`, `sim_net/env_handler.py`, `replay.py`.
- **Tests:** headless TCP-step recording test; replay unit test.
- **Acceptance:** `--headless` produces frames for external steps.

## CP-08 — Sensor depth

- **Requirements:** PH0-SEN-007 residual, PH75-SEN-009 (B-P2-1 class)
- **Items:** camera near/far/roll; procedural fallback honors pose+FOV; PiP overflow guard.
- **Files:** `camera_sensor.py`, `sensor_config.py`, `app.py` PiP.
- **Validation:** rear camera headless output verified; 4-cam smoke.

## CP-09 — UI dead controls + capabilities surface

- **Requirements:** PH1-UI-010, PH3-DSN-003, PH3-UI-011, PH7-DAT-008 (B-M4, B-M11)
- **Items:** wire-or-remove scen_*/act_min/max; emit sp_*/vc_* rows incl. spawn yaw; dataset export action in DATA; throttle scans.
- **Files:** `inspector.py`, `datasets_panel.py`.
- **Validation:** smoke checks per control.

## CP-10 — Training evidence artifacts

- **Requirements:** PH5-CUR-001, PH5-BAT-007, PH5-WRK-009, PH5-DAT-011, PH6-BEN-007 (B-H6-adjacent)
- **Items:** curriculum run; real batch_result; remote-worker two-process run; exported+validated dataset; learn-bench results/.
- **Deps:** CP-06 (reuse the working trainer). **Acceptance:** artifacts on disk.

## CP-11 — Packaging phase

- **Requirements:** PH0-PKG-018 (B-C2) — per GAP-02. **Deps:** CP-01..10 preferred to avoid re-verification.
- **Acceptance:** dist launches clean-dir; user data outside _internal; training policy decided + verified.

## CP-12 — Theme/accessibility polish

- **Requirements:** PH7-THM-002 + B-L* (B-M9, B-L3..L6, L10)
- **Items:** tokenize HUD/editor; ui_scale metrics; keyboard nav; bounded text; terminology; PiP overflow; SCENE cap; dead-code prune; doc refresh (test counts, perf numbers).
- **Acceptance:** light-theme screenshots all tabs; bounded-text pass.

---

# REPAIR-PHASE UPDATE (2026-10-04)

Stages 1–6 of the staged repair plan executed and verified
(see `docs/REPAIR_IMPLEMENTATION_REPORT.md`). Stage 7 regression: 550
tests passing. Open work: E002–E010 full-scale matrix + converged
safe-driving (ARL-009), 4-wheel/Pacejka envelope (PH0-VEH-005),
editor gizmos (PH2-ED-002), UI polish rows (PH4-UI-014, PH6-UI-011,
PH7-REC-006, PH7-DAT-008).
