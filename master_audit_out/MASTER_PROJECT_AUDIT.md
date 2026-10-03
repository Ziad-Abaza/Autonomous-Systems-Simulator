---
noteId: "40128470bf6a11f1a29f1fbaabbd87c8"
tags: []

---

# MASTER PROJECT AUDIT

Full requirement-by-requirement traceability audit of the autonomous-driving
simulation platform against `docs/MASTER_PROJECT_PLAN_BASELINE.md`.

**Audited:** `D:\coding\projects\Simulation` @ `feat/agentrl` HEAD `f177909`
(the repo was a moving target — a parallel agentRL session committed T7–T13 and
modified `.gitignore` during the audit; all findings timestamped 2026-10-03).
**Companion docs:** `MASTER_REQUIREMENTS_TRACEABILITY.md`, `MASTER_PROJECT_STATUS.md`,
`MASTER_BLOCKERS.md`, `MASTER_GAP_ANALYSIS.md`, `MASTER_COMPLETION_PLAN.md`,
`master_audit/BASELINE_CORRECTIONS.md`.
**Evidence package:** `docs/master_audit/evidence/` (screenshots ×93, runtime
scripts + outputs, real training artifacts, vehicle-dynamics rerun).

# Executive Summary

136 requirements audited: **103 COMPLETE · 27 PARTIAL · 4 IMPLEMENTED-BUT-BROKEN ·
2 IMPLEMENTED-BUT-UNVERIFIED · 0 DOC-ONLY · 0 STUB · 0 MISSING · 5 SUPERSEDED.**

This is a substantially functional platform: 492/492 tests, a real PPO run and a
real SAC run executed during the audit, the full experiment lifecycle exercised via
CLI, 17 track files loaded + simulated, 31/31 UI smoke checks, 15/15 TCP e2e, and a
freshly re-run vehicle-dynamics validation suite. But two structural findings
dominate:

1. **It has never produced a learning result.** Every training artifact — including
   a 4,096-step PPO run executed *during this audit* — yields a stationary policy
   (mean_speed ~0.01 m/s, ~0.79/step idle reward vs 0.83/step driving). The reward
   permits idling. The agentRL phase is the correct in-progress fix (T1–T13 landed).
2. **Standalone packaging — an original headline requirement — is broken.** No dist,
   no `_MEIPASS` handling, data→`_internal/`, torch excluded while UI imports it.

Second-tier findings are contract-level: HANDSHAKE serves legacy spaces for authored
agents; step-after-done returns both done-flags; open tracks have no end-of-course
semantics (`is_on_road=True` past the finish — measured); all templates ship with
completion termination disabled; a disabled sensor leaves an orphaned observation
channel filled with constants (validator catches it, runtime tolerates it).

# Original Plan Reconstruction

Ten phases recovered (see baseline doc): P0 platform → P1 hardening → P2 authoring →
P3 env designer → P4 experiments → P5 scale-out → P6 scalable training → P7 studio UX
→ P7.5 repair → VD dynamics → agentRL. Two supersessions documented. Discrepancies
between baseline and reality recorded in `BASELINE_CORRECTIONS.md` (8 items; most
notable: "4-wheel model" was always axle-aggregate; PKG-018 only verified in Phase-1
era).

# Verification Methodology

Every requirement traced: implementation → tests → runtime. Claims require A/B/C
evidence; D/E never sufficient for COMPLETE. Fresh runtime evidence generated this
audit: vehicle-validation suite (12 maneuvers), 7-template drive matrix (1,800 steps
each), open-track overshoot repro, idle-vs-drive reward probe, sensor-disable +
validator probe, deterministic 200-step bitwise reproduction, env.step throughput,
experiment lifecycle (validate-env→create→launch→COMPLETED), real PPO training run,
real agentRL SAC training run, ui_shots (93 screenshots, spot-verified visually),
smoke_interactions (31/31) + tcp_recording_e2e (15/15) re-runs, full suite re-run
(492 pass) + tests/agent (67/68, 1 in-flight failure).

# Requirement Traceability

See `MASTER_REQUIREMENTS_TRACEABILITY.md` — one row per requirement, all 136.

# Phase 0 Status

13 COMPLETE / 4 PARTIAL / 2 BROKEN. Core sim, physics, spline, mesh, sensors,
reward/termination, protocol, persistence, UI, editor all verified. BROKEN:
PH0-PKG-018 (packaging) and PH0-NET-013 (adapter contract). PARTIAL: PH0-VEH-005
(spec overclaims 4-wheel; delivered single-track), PH0-ENV-011 (dead scenario fields),
PH0-ENV-010 (dead global_seed), PH0-REC-015 (headless gap + replay fidelity).

# Phase 1 Status

6 COMPLETE / 2 PARTIAL / 2 BROKEN. Steering/PID/sanitization/leakage/checkpoint
guards done. BROKEN: PH1-ENV-007 (same contract defect), PH1-TRK-009 (completion
mechanism exists but unusable — off in all templates + no end-of-course boundary).

# Phase 2 Status

8 COMPLETE / 3 PARTIAL. Entities, broadphase, schema 2.0, vision, PPO, docs done.
PARTIAL: ED-002 (viz gizmos missing), ED-004 (8-entity cap), RL-010 (real PPO,
artifacts exist, outcome exposes reward exploit — fresh evidence this audit).

# Phase 3 Status

10 COMPLETE / 4 PARTIAL. All designers + compiled pipelines + leakage gate +
validator + DISCOVER_CONTRACT + schema 3.0 verified. PARTIAL: DSN-003 (dead
act_min/max rows), DSN-006 (dead scenario fields + swallowed errors + target_speed
leak), VAL-009 (diff coverage), UI-011 (dead controls).

# Phase 4 Status

12 COMPLETE / 4 PARTIAL. Entire experiment platform verified — **executed live
this audit**: `validate-env` → `create` → `launch` → COMPLETED run with contract/
metrics/checkpoints/run_result/artifacts. PARTIAL: BAT-010 (`cli batch` dead),
CLI-013, UI-014 (scan perf + export reachability), RUN-004 (orphan-on-restart).

# Phase 5 Status

12 COMPLETE / 1 UNVERIFIED. Curriculum runtime, capabilities, harness, SAC, DQN,
scheduler, local workers, analytics, dataset tools all real + tested. UNVERIFIED:
PH5-WRK-009 remote workers — real implementation, zero artifacts.

# Phase 6 Status

9 COMPLETE / 3 PARTIAL. VectorEnvs, SET_SCENARIO, tcp_multi, dataset tools, BC,
worker hardening, measured perf — all verified. PARTIAL: VEC-006 (dirty-env
recycle), BEN-007 (never executed — no results/), UI-011 (read-only rows).

# Phase 7 Status

6 COMPLETE / 4 PARTIAL. Widgets, library, shell, editor, replay, dialogs, smoke —
verified. PARTIAL: THM-002 (HUD/editor hardcoded → dark on light themes), REC-006
(headless + persistence), DAT-008 (scans + export), QA-010 (exit-code bug).

# Phase 7.5 Status

10 COMPLETE / 2 PARTIAL. All repair findings verified fixed incl. multi-camera +
SENSORS tab (screenshots + smoke). PARTIAL: SEN-007 (disabled-sensor orphan
channel — new finding), TPL-012 (all templates ship `term_completion` off — new
finding).

# Vehicle Dynamics Status

10/10 COMPLETE within stated envelope. Fresh suite rerun: mild-steer sideslip 3.77°
(vs 8.2° pre-fix); deterministic; substeps; honest envelope 0–30 m/s, mu 0.6–1.2.
Caveats: at-limit maneuvers show 34.9°/88.6° sideslip (plausible trail-brake physics,
not reference-validated); detection-only collision; no post-peak tire.

# agentRL Status

7 COMPLETE / 1 PARTIAL / 1 UNVERIFIED — assessed mid-flight. Committed T1–T13:
foundations, obs/action, rewards, envs, replay, SAC, PPO, trainer+checkpoints,
eval suite, ContinualTrainer, lab scripts + TCP e2e. A real 1,500-step SAC run
executed this audit produced checkpoints + metrics and demonstrated the anti-exploit
reward punishing idling (−0.02/step). UNVERIFIED: ContinualTrainer (T12) — machinery
tested, no real A→B→C run. PARTIAL: T14–T15 — `experiments/matrix.py` committed
(`99d3083`) with E001–E010 configs; **E001 (60k-step oval, SAC+PPO) ran during this
audit and produced the first-ever movement signal**: early episodes terminate `stuck`
at 1,200 steps (mean_speed ~8e-6 m/s, return −23.97); by ~15k steps episodes
terminate `collision` at ~190 steps with mean_speed ~1.9–2.0 m/s and returns >+10.
The anti-exploit reward demonstrably broke the stationary optimum. The final eval
gate (mean_speed>2 m/s) is still pending — the run was in progress at audit time.

# Cross-System Integration Status

| Integration | Verified? | Evidence |
|---|---|---|
| Editor → project → env | YES | smoke + template matrix |
| Sensors → obs pipeline → vector | YES | live obs (23,), channel list |
| Sensor disable → obs contract | PARTIAL | orphaned channel w/ constant fill (B-H5) |
| Env → TCP → client | YES | e2e 15/15 |
| TCP → authored-agent contract | NO | B-H1 |
| UI → experiment → trainer → artifacts | YES | live CLI + TRAIN providers |
| Curriculum → trainer | B-tested only | no run artifact |
| Scheduler → remote worker | UNVERIFIED | no artifacts |
| Recording → replay → UI | YES (GUI) | e2e 15/15; headless gap B-H4 |
| agentRL → sim public API | YES | envs/factory uses public API only; real SAC run |

# Critical Findings

1. Stationary-policy reward exploit — measured live (idle 0.79/step ≈ driving 0.83/step); fresh PPO run reproduces a motionless policy end-to-end. **During this audit, agentRL E001 produced the first counter-evidence**: the anti-exploit reward moved a SAC agent from `stuck`-stationary episodes to driving-until-collision at ~2 m/s.
2. Packaging structurally broken (B-C2).
3. Contract defects: HANDSHAKE legacy spaces; step-after-done both-flags.
4. Open-track semantics: no end boundary + completion disabled everywhere.
5. Disabled-sensor orphaned channels silently feed constants to agents.

# Broken Features

PH0-PKG-018 packaging · PH0-NET-013/PH1-ENV-007 adapter contract · PH1-TRK-009
open-track completion (config-level) · `cli batch` · dead inspector controls ·
headless recording · smoke tools exit codes · test_experiment_matrix (in-flight).

# Missing Features

Completed E001 eval verdict + E002–E010 · remote-worker/batch/curriculum/dataset/learn-bench
artifacts · camera near/far/roll + pose-aware fallback · spawn param editing ·
sensor preview + plugin registry · replay/packaging/dataset-scan tests · keyboard
nav · theme coverage in HUD/editor.

# Unverified Features

Remote workers (PH5-WRK-009) · ContinualTrainer end-to-end (ARL-007) · BC production
run · resume-equivalence deep check · packaged build (stale E evidence) ·
experiment-level bitwise training determinism (env-level verified).

# Technical Debt

Dual-path environment (~150 lines + key/reason drift) · dead controls/handlers ·
`tools/` import from sim_ui (packaging + layering) · orphan scripts + legacy dirs in
experiments/ · presets dual-source · global RNG pollution in trainers · unbounded
buffers/no timeouts · ppo_train eager torch import · stale docs numbers.

# Evidence Index

`docs/master_audit/evidence/` — see `INDEX.md`:
- `screenshots/` ×93 (fresh ui_shots run, incl. multi-camera SENSORS tab, light theme, FINISH marker)
- `agent/sac_smoke/` real SAC run (metrics.jsonl + 3 checkpoints)
- `experiments/exp_38eba258e68bf064` real PPO run (contract/metrics/ckpts/run_result) + `exp_b87af2db930f3dcc` dummy-run lifecycle
- `vehicle/audit_run/` fresh 12-maneuver validation
- `tracks/` template matrix + overshoot repro
- `tests/` suite outputs (492 + 67/68)
- `scripts/` reproducible audit scripts (`_ev_*`)
- performance: env.step ~2,281/s inprocess; PPO train ~250 sps; SAC ~68 sps; reset ~4 ms

# Completion Plan

`MASTER_COMPLETION_PLAN.md`: CP-01..05 contract hardening → **CP-06 agentRL T14–T15
+ E001 (next)** → CP-07..10 recording/sensors/UI/training-artifacts → CP-11 packaging
→ CP-12 polish.

# Completion Gates (define "project complete")

| Gate | Pass? | Evidence needed |
|---|---|---|
| Core sim verified | YES | tests + runtime |
| Authoring: create/save/reopen/run | YES | smoke + matrix |
| Dynamics validated | YES (envelope) | VD suite rerun |
| Sensors configurable+integrated | PARTIAL | near/far/roll + orphan fix |
| RL env/protocol contract-correct | PARTIAL | CP-01/02 fixes |
| External agent trains+evaluates | PARTIAL | runs execute; learning unproven |
| Experiments reproducible+resumable | PARTIAL | machinery yes; training-level bitwise unproven |
| **Continual learning demonstrated** | **NO** | A→B→C retention numbers |
| **Generalization measured** | **NO** | unseen-track eval artifacts |
| Standalone distribution | NO | packaging phase |

# Final Readiness Assessment

| Question | Answer |
|---|---|
| Install + launch? | YES — `main.py` boots, smoke-tested (dev env only) |
| Create a complete track? | YES — editor + templates + validation (screenshots) |
| Open AND closed tracks? | PARTIALLY — both authorable; open tracks can't "finish" (B-H2) |
| Configure entities? | YES — place/drag/edit verified |
| Configure sensors? | YES — SENSORS tab end-to-end |
| Configure cameras? | PARTIALLY — multi-cam verified; no near/far/roll; fallback ignores pose |
| Valid observations? | YES — governed pipeline + leakage gate (orphan-channel caveat) |
| External AI controls vehicle? | YES — TCP lockstep verified; contract defect for authored agents |
| Dynamics sufficiently validated? | PARTIALLY — rigorous in envelope; not research-grade beyond it |
| Run RL training? | YES — real PPO + SAC runs executed this audit |
| Reproduce experiments? | PARTIALLY — config/fingerprint machinery + env determinism verified; training-level bitwise unproven |
| Resume from checkpoints? | PARTIALLY — implemented + tested mechanically; resume-fidelity not deep-verified |
| Concurrent environments? | YES — measured 1,057 sps @4 envs (process) |
| Generate datasets? | PARTIALLY — tools verified; no artifacts on disk |
| Replay trajectories? | PARTIALLY — works; fidelity + headless gaps |
| Learn across multiple tracks? | UNVERIFIED — ContinualTrainer tested mechanically; no A→B→C artifact |
| Retain previous knowledge? | NO — no retention run exists |
| Learn obstacle avoidance? | NO — scenario mutator exists; no training proof |
| Generalize to unseen tracks? | NO — eval matrix exists; never run |
| Diagnose failures? | PARTIALLY — failure classifier + termination reasons exist; not exercised on real runs |
| Distribute standalone? | **NO** — packaging broken |
