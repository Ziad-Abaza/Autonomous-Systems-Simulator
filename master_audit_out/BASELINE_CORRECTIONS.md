---
noteId: "ab6ffaf0bf6911f1a29f1fbaabbd87c8"
tags: []

---

# BASELINE CORRECTIONS

Discrepancies found between `docs/MASTER_PROJECT_PLAN_BASELINE.md` and reality during
the traceability audit. The baseline was NOT modified; corrections recorded here.

## C-1 — PH0-VEH-005 overclaims "four-wheel dynamic model"

- **Baseline says:** "Four-wheel dynamic vehicle model with nonlinear tire slip" (faithful to the original README).
- **Reality:** `sim_core/vehicle_model.py` implements a **single-track (bicycle) axle-aggregate** model — front/rear axle tires, not four per-wheel tires. `VEHICLE_DYNAMICS_MODEL.md` documents this honestly.
- **Correction:** the original spec's "four-wheel" wording was never literally implemented at any commit; treat as axle-pair dynamic model. If per-wheel fidelity is ever required, that is new scope.

## C-2 — PH1-TRK-009 is mechanism-complete but configuration-broken

- **Baseline says:** "Open-track course completion support" — Phase 1 added it.
- **Reality:** the mechanism exists (`termination_designer.py:200`) and is tested, BUT every shipped template and preset ships `term_completion enabled:false` (verified programmatically across all 7 templates + presets). Additionally, `TrackSpline.get_closest_point` clamps to the nearest sample, so beyond an open end the lateral offset is measured against the terminal tangent — driving straight past the finish reads ~0 lateral offset → `is_on_road=True` indefinitely (`empty` and `slalom` templates ran 1,800 steps "running"; `tracks/untitled` reached x=103.9 on a 100 m route).
- **Correction:** requirement should read "open-track course completion mechanism + shipped configuration that can actually finish". Currently the latter fails.

## C-3 — PH0-PKG-018 was only ever verified in the Phase-1 era

- **Baseline says:** standalone PyInstaller executable — verified.
- **Reality:** the Phase-1 audit verified a then-current build. The current source cannot produce a correct distribution: no `_MEIPASS` handling, user data resolves into `_internal/`, `torch` excluded while UI training imports it, `tools/` imported by `sim_ui` isn't packaged, `datas=[]` empty. No `dist/` artifact exists.
- **Correction:** mark as "implemented then; broken now" — a packaging remediation phase is required.

## C-4 — PH0-NET-014 "PPO agent" was never the real algorithm originally

- **Baseline says:** reference agents incl. PPO (per README).
- **Reality:** `ppo_train.py` was a rollout harness with a fixed 24-dim obs assumption; the real PPO (`ppo_baseline.py`) arrived in Phase 2. Both coexist; `ppo_train.py`'s eager torch import also breaks `import sim_client.agents` without torch.
- **Correction:** classify ppo_train as superseded legacy; PPO requirement fulfilled by ppo_baseline + contract trainers.

## C-5 — ARL ledger staleness vs actual commit state

- **Baseline says:** ARL-003..009 tasks T7–T15.
- **Reality at audit time:** T1–T13 are committed (commits `c680176`–`f177909`: replay buffers, SAC, PPO, trainer+ckpts, eval suite, ContinualTrainer, lab scripts + TCP e2e). T14 is in flight with a **failing** test (`test_experiment_matrix.py::test_experiment_config_loads` — `KeyError 'algo'`); T15/E001 not run. `tests/agent` = 68 tests, 67 pass.
- **Correction:** audit statuses reflect commit reality, not plan checkboxes.

## C-6 — `compute_steps_for_real_frame` dead code

- **Baseline implies:** decoupled render vs physics timestep (Phase-1 time model).
- **Reality:** `FixedClock.compute_steps_for_real_frame` exists but is dead — the env only uses `advance_fixed_step` (always 60 Hz regardless of wall dt).
- **Correction:** the "decoupled" claim holds conceptually (net steps are lockstep; render is view-only) but the wall-clock path is unused.

## C-7 — PH2-ED-002 partially reinterpreted

- **Baseline lists:** "control-point insertion; width handles; banking/elevation cues; curvature visualization" (Phase-2 plan wording).
- **Reality:** insertion + drag + POINT-tab editing exist (width/elevation/banking/friction are real `ControlPoint` fields); curvature/bank/tangent visualization gizmos were never built.
- **Correction:** split into (a) point manipulation — done; (b) visualization aids — missing.

## C-8 — Sensor disable semantics

- **Not in baseline explicitly.** Discovered: disabling a `sensor_config` removes the sensor from the suite but leaves the obs channel orphaned — runtime fills it with constant `[1.0]*15`; the validator correctly emits ERROR (`lidar_ranges requires sensor lidar_rays`). So the safety net exists as validation but the runtime tolerates the invalid state with a silent default fill.
