---
noteId: "c9dc5e10bf7411f1a29f1fbaabbd87c8"
tags: []

---

# Vehicle Dynamics Repair Report

**Date:** 2026-10-04
**Scope:** post-`7481f50` control degradation — "vehicle significantly harder to
control" after the tire-stiffness/friction-ellipse/load-transfer rework.
**Method:** root-cause review of the control chain + model, deterministic
maneuver evidence, no stabilization cheats.

---

## 1. Current Problem

After commit `7481f50` ("fix excessive drift via corrected tire stiffness,
friction ellipse, load transfer, substepping") the vehicle became
disproportionately hard to control:

- a small steering input under moderate/high throttle produces a growing,
unrecoverable power-oversteer slide — measured: full-throttle launch then
  steer cmd 0.10 (3.3° road wheel) at ~13 m/s → sideslip grows 0 → 60°+,
  yaw rate −88°/s, spin;
- a steady 15 m/s corner (~0.35 g lateral demand) + full throttle →
  sideslip 24.8° and diverging → spin;
- lateral/track control feels unstable specifically whenever significant
  throttle or brake is applied — pure coasting and gentle inputs were fine.

## 2. Regression Introduced by Latest Changes

`7481f50` was itself a correct and needed fix (verified in
`VEHICLE_DYNAMICS_AUDIT.md`): cornering stiffness ~6× too soft produced
8–25° sideslip in ordinary cornering. It also introduced, for the first
time, a **combined-slip friction envelope** — previously lateral and
longitudinal forces saturated *independently*, so throttle application had
zero effect on cornering grip.

The regression is in **how the envelope allocates grip**, compounded by a
**powertrain parameter that was never traction-feasible**:

| Defect | Where | Effect |
|---|---|---|
| Demand-first ellipse: `u = Fx_demand/(µ·Fz)`; `lat_cap = µFz·√(1−u²)` | `vehicle_model.py` | The tire is assumed to *deliver* all demanded longitudinal force up to µ·Fz and keep only the ellipse remainder laterally. Demanding force ≠ achievable force — excess demand should become wheelspin/slide, not consume lateral grip. |
| `max_drive_force = 6500 N`, all through the rear axle (RWD) | `vehicle_config.py` | Rear axle static cap is only µ·Fz_r ≈ 5650 N. Full throttle therefore pegs `u_r → 1` at all speeds below ~25 m/s → rear lateral capacity collapses toward **0** → snap oversteer under power. |
| Linear force taper `F ∝ (1 − v/v_top)` | `vehicle_model.py` | Artificial curve: under-delivers mid-range thrust and **amplifies** force in reverse (factor > 1 for v < 0). |

Before `7481f50` this was masked: with no envelope at all, the rear axle
always had full lateral force regardless of throttle.

## 3. Root Cause

**Primary:** the friction-envelope allocation model is wrong. A contact
patch produces a bounded force *vector* whose direction follows the
combined-slip state; it cannot be commanded to deliver 6500 N forward and
still supply the demanded lateral force. The demand-first ellipse treats
demanded Fx as delivered Fx, exhausting the circle longitudinally and
zeroing lateral capacity exactly when the driver needs it most (power
application).

**Secondary (parameter):** 6500 N of rear-wheel drive demand on a
~5650 N rear friction cap guarantees `u → 1` under full throttle — even a
correct envelope spends most of it longitudinally. This value was chosen
when the model had no envelope; it was never transmissible.

**Not the cause (verified):** steering sign/rate/limits, slip-angle signs,
low-speed blend, substep count, load transfer, proper-acceleration
bookkeeping — all reviewed stage-by-stage (§5) and individually sound.

## 4. Physics Model Review

Full review of `vehicle_model.py` / `vehicle_config.py`:

- **Geometry/loads:** `a = L(1−wf)`, `b = L·wf`; static split `mg·b/L`,
  `mg·a/L`; longitudinal transfer `m·ax_proper·h_cg/L` clamped to ±90% —
  correct, uses proper acceleration (the quantity that transfers load).
- **Slip angles:** `atan2(vy ± a,b·wz, vx_eff)`, signed deadzone ±0.5 —
  correct, reverse-correct.
- **Tire curve:** saturating tanh, `Ca/µFz ≈ 13–15 /rad` — representative;
  monotonic (no post-peak falloff) — documented limitation, benign.
- **Friction envelope:** *replaced* — see §7.
- **Drive force:** *replaced* — power-limited `min(F_max, P/v)` + limiter
  band — see §7.
- **Brakes:** 9000 N total, 65% front, per-axle envelope. Hard
  brake + steer spins (non-ABS behavior) — physical, documented.
- **Integration:** explicit Euler, 240 Hz internal substeps — stability
  margin verified analytically (worst-case tire stiffness eigenvalue
  h·λ ≈ 0.57 inside the blend band) and empirically (§10).
- **Low-speed blend:** bounded relaxation τ = 0.05 s to kinematic bicycle —
  bounded, stable.
- **Collision:** detection-only (SAT) — out of dynamics scope, unchanged.

## 5. Control Chain Review

```
action [-1,1]/[0,1] → decode (clip, NaN→0) → (steer, thr, brake)
  → target_steer = −steer·0.58 rad        (sign: cmd<0 → +δ → left → +yaw) ✓
  → rate limit 4.5 rad/s                   ✓ (fast, but physical)
  → α_f = atan2(vy + a·wz, vx_eff) − δ     ✓
  → α_r = atan2(vy − b·wz, vx_eff)         ✓
  → per-axle (Fx_i, Fy_i) → envelope       ← FIXED (was demand-first)
  → ay = (Fy_f·cosδ + Fy_r)/m − vx·wz      ✓
  → az = (a·Fy_f·cosδ − b·Fy_r)/Iz         ✓
  → Euler @240 Hz → yaw → world pose       ✓
  → track queries (s, lateral, heading)    ✓
```

Units, signs, coordinate conventions verified consistent end-to-end;
the only broken stage was grip allocation under combined demand.

## 6. Drift/Sideslip Analysis

Requirements: stable cornering, mild controllable sideslip, larger slip
near the limit, progressive loss, recovery on input reduction, provoked
drift possible. Result after repair (evidence §9):

- gentle cornering: β ≈ 3.5° at 0.44 g — planted;
- near-limit cornering: transient slides to ~29° then converges to a
  steady −1.6° corner after speed scrubs — progressive, not a cliff;
- power oversteer: sustained full throttle mid-corner holds a stable
  drift attitude (~57°) that **recovers on throttle lift** (β → ~3°) —
  physical RWD behavior;
- provoked drift: 0.8 lock + full throttle from 8 m/s → 26–68° slides,
  hands-off recovery → β → 0°;
- brake + steer (0.8 brake) → spin-to-stop: real non-ABS limit departure.

Drift is a consequence of the friction envelope, not scripted; the model
was not artificially stabilized.

## 7. Changes Made

**`sim_core/vehicle/vehicle_model.py`**

1. Friction envelope → **proportional resultant clamp** per axle:
   compute free-rolling lateral demand `Fy_i = −cap·tanh(Ca·α/cap)`,
   then `s_i = min(1, cap_i/|F_i|)` applied to the whole (Fx, Fy) vector.
   No Fx-priority: excess demand becomes slide, preserving the demanded
   force direction. Envelope invariant `|(Fx,Fy)| ≤ µ·Fz` still asserted
   by tests. Kinematic branch also clamps Fx to the axle cap.
2. Drive force → **power-limited engine**: `F = min(F_max, P/|v|)`
   (`P = engine_power`, floored at `power_min_speed`), plus a smooth
   2 m/s limiter band into `top_speed` (forward). Removes the linear
   taper's mid-range under-delivery and its >1× amplification in reverse.

**`sim_core/vehicle/vehicle_config.py`**

3. `max_drive_force` 6500 → **5000 N** — ≈0.88× the rear axle's static
   friction cap (~5650 N); the most a no-diff/no-TC RWD can transmit
   without guaranteed wheelspin.
4. New: `engine_power = 90 000 W` (~120 hp generic compact),
   `power_min_speed = 2.0 m/s`. `top_speed` role clarified: speed-limiter
   band edge.

**`tools/vehicle_dynamics/maneuvers.py`** — +5 maneuvers: `s_curve`,
`throttle_in_corner`, `drift_initiation`, `drift_recovery`,
`full_throttle_steer_correction` (regression maneuver for this bug).
Suite now 17 maneuvers.

**`tests/test_vehicle_dynamics_validation.py`** — +`TestCombinedSlipAllocation`
(3 tests): full-throttle small-steer no-spin, throttle-in-corner bounded +
recovery, speed limiter respected.

## 8. Parameters Changed and Physical Justification

| Param | Old | New | Justification |
|---|---|---|---|
| `max_drive_force` | 6500 N | 5000 N | ≈0.88× rear static traction (5650 N); a RWD axle with no TC/diff cannot usefully exceed its friction cap — it wheelspins. Generic 1200 kg passenger-car thrust. |
| `engine_power` | — | 90 kW | ~120 hp generic compact; replaces artificial taper with the physical `F = P/v` bound. |
| `power_min_speed` | — | 2.0 m/s | Guards 1/v at crawl; F_max already caps below it. |
| `top_speed` | taper endpoint | limiter band | Same 45 m/s value; now a speed-limiter fade instead of a linear force taper to zero. |

No parameters were weakened to hide the problem: tire friction, cornering
stiffness, steering limits/rate, inertia, and the envelope bound are
unchanged in kind. The two model changes repair an incorrect allocation
rule and an artificial drive curve.

## 9. Maneuver Results

Suite: `benchmarks/vehicle_dynamics/repaired/` (results.json + 17 CSVs +
plots; regenerate: `python tools/vehicle_dynamics/run_validation.py --tag repaired`).
20 m/s cruise, µ = 1.0.

| Maneuver (V-map) | peak β | steady/final β | Verdict |
|---|---|---|---|
| V01 straight accel / V02 cruise | 0.00° | 0.0° | pass |
| V04 const steer 0.10 | 3.53° | steady ~3.7° | planted |
| V04 const steer 0.25 | 29.4° transient | settles −1.6° | convergent slide |
| steer step 0.15 | 4.37° | 4.1° | crisp, bounded |
| steer ramp 0.30 | 22.9° | converges | progressive |
| V05 sine 0.15/0.5 Hz | 5.65° | — | pass |
| steer reversal 0.20 | 19.8° | −17.1° (opposite corner) | bounded |
| V10 escalation → cmd 1.0 | 38.2° | slides at limit | expected at 2 g+ demand |
| V03 braking | 0.00° | 0 | pass |
| V09 brake+steer 0.8 | 88.6° | 0 (stopped) | non-ABS spin — physical |
| V06 lane change @20 m/s | 22.6° | sustained slide | ~2.3 g demand > envelope — real departure |
| V13 disturbance kick | 2.73° | recovers, wz→~0 | pass |
| V07 S-curve | 18.8° | 0.0° | pass |
| V08 throttle-in-corner | sust. drift while held | 3.3° after lift | bounded, recoverable |
| V11 drift initiation | 68° peak | — | provocable |
| V12 drift recovery | 35.2° peak | 0.0° | full recovery |
| **full-throttle steer corr.** | **3.18°** | — | **was: spin >60° — regression fixed** |

## 10. Timestep Results

Same scripted scenarios at caller dt 30/60/120/240 Hz, final-pose delta
vs 240 Hz reference (internal 240 Hz substeps):

| Scenario | max pos delta | max yaw delta |
|---|---|---|
| mild corner | 0.23 m | 0.19° |
| throttle-in-corner | 0.61 m | 0.11° |
| sine steer | 0.05 m | 0.30° |
| drift provoke/recover | 0.21 m | 0.69° |

Qualitative behavior is consistent; no integration instability.
(Script: `_tmp_view/dt_check.py`.)

## 11. Determinism Results

Identical scripted runs (incl. the drift scenario) reproduce bitwise-equal
final state — `determinism bitwise equal: True`; also asserted by
`TestDeterminism` (existing regression test).

## 12. RL Impact

Frozen **PD baseline driver** through `agentRL.eval.evaluate_policy`,
3 seeds × 3 tracks, 1500-step horizon — before vs repaired
(old model loaded from git HEAD for a true A/B):

| Track | Metric | Before | Repaired |
|---|---|---|---|
| serpentine | mean_return | 562.7 | **656.7** |
| | mean_speed | 6.70 | **7.63** |
| | mean_lateral | 1.08 | 1.02 |
| | mean_heading_err | 0.449 | **0.338** |
| | collision/off-road | 0 / 0 | 0 / 0 |
| oval | mean_return | 385.0 | **453.0** |
| | mean_heading_err | 0.598 | **0.542** |
| gen_loop_0 | mean_return | 358.4 | **444.6** |
| | mean_heading_err | 0.631 | **0.550** |

Steer smoothness ≈0.997 both. No regression on any metric; speed/heading
improve because the repaired powertrain can apply throttle without
collapsing lateral grip. The conservative PD driver masks the old bug
(modest throttle); the maneuver suite shows where it actually hurt —
RL agents exploring high-throttle actions and WASD users hit it directly.

Observation schema, action bounds, reward inputs unchanged.

## 13. Remaining Limitations

- Sustained full-throttle mid-corner holds a stable drift (physical for
  RWD, recoverable on lift) — agents can still learn to provoke slides.
- Hard brake + steer spins (no ABS model) — physical, flagged in tests.
- Lane-change-level inputs (>~1 g demand) legitimately depart; there is
  no ESC safety net.
- tanh tire has no post-peak falloff; no per-wheel load transfer
  (axle-aggregate); no suspension/roll; detection-only collision;
  planar z (elevation is tracked for rendering, not integrated).
- Envelope is a resultant-vector clamp — not a Pacejka combined-slip
  curve; direction preservation is an approximation near the limit.
- Not validated: µ < 0.5, speeds > ~35 m/s.

## 14. Evidence Index

- Code: `sim_core/vehicle/vehicle_model.py` (envelope, drive force),
  `sim_core/vehicle/vehicle_config.py` (params).
- Suite output: `benchmarks/vehicle_dynamics/repaired/` (results.json,
  17 CSVs, plots, manifest); compare vs `baseline/` and `fixed/`.
- Tests: `tests/test_vehicle_dynamics_validation.py` (16 tests,
  incl. new `TestCombinedSlipAllocation`), `tests/test_vehicle_physics.py`.
- Scripts: `_tmp_view/dt_check.py` (timestep + determinism),
  `_tmp_view/rl_baseline{,_before}.py` + `.json` (PD eval A/B),
  `_tmp_view/env_check.py` (env-level drive).
- Prior docs: `docs/VEHICLE_DYNAMICS_AUDIT.md`,
  `docs/VEHICLE_DYNAMICS_MODEL.md`, `docs/VEHICLE_DYNAMICS_VALIDATION_REPORT.md`.
