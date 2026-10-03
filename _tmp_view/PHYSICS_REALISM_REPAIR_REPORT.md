---
noteId: "560adba0bf7f11f1a29f1fbaabbd87c8"
tags: []

---

# Physics Realism Repair Report

Date: 2026-10-04 · Simulator: Simulation (single-track vehicle model, 60 Hz
env / 240 Hz physics) · Scope: vehicle dynamics, road/surface physics,
collision/contact, sensor consistency, validation suite.

> Procedures in this report are **inspired by / aligned with the
> methodology of** SAE J266, ISO 4138, ISO 7401 and ISO 3888-1. The
> simulator is not certified against these standards; the metrics they
> define (understeer gradient, response delay, overshoot) are computed on
> the model to enable quantitative comparison.

---

## 1. Executive Summary

The reported high-speed failure (excessive sideslip, lost grip,
uncontrollable recovery) traced to **one code defect** and **two missing
physics paths**, layered on top of physically valid limit behavior:

- **R1 — lateral-force freeze.** The low-speed kinematic blend was keyed
  on `|vx|` only. A high-speed slide that scrubbed forward speed below
  1.5 m/s switched off *all* lateral tire force and yaw dynamics: the car
  glided sideways at ~18–20 m/s for 15+ s and drifted ~175 m laterally.
  This is the "grip disappears / unrecoverable" symptom.
- **R2 — no self-aligning mechanism.** The monotonic saturating tire with
  no relaxation lag and no aligning moment produced ~zero net yaw torque
  in saturated slides — a released slide could persist. Real tires have
  relaxation length (transient lag) and pneumatic trail (aligning
  moment); both were added.
- **R3 — environment physics dead ends.** `ControlPoint.friction` was
  authored and serialized but **dropped by the spline builder** —
  per-point friction never reached the tires. `banking` propagated to
  samples but was never consumed. Collision was detection-only: barriers
  flagged `is_colliding` but did not impede the vehicle.

After repair: hands-off release from a 40 m/s saturated slide
self-recovers in ~0.8 s (previously irrecoverable); countersteer recovery
0.35 s; sub-limit cornering at 44 m/s holds 3–4° sideslip at ~0.87
utilization; measured understeer gradient +2.1 °/g; collision removes
98% of impact KE without tunneling.

---

## 2. Observed High-Speed Failure

Reproduced with `_tmp_view/vy_freeze.py` (pre-fix state):

```
t    vx      vy      wz(deg)
1.5   0.54   24.85   -19.8      <- vx below low-speed threshold
2.0   0.30   23.69    -0.0      <- yaw dynamics switched OFF
14.5  0.05   18.27    -0.0      <- still gliding laterally
final: vx=0.05 vy=18.11, drifted to x=334.3 y=-175.0
```

A 40 m/s slide decayed `vx` through the 1.5 m/s threshold while `vy`
remained ~20 m/s. The kinematic branch then forced `Fy=0`, `wz→0`, and
only the small rolling resistance on `vx` remained — the car translated
laterally nearly undamped.

## 3. Root Cause

`vehicle_model._substep` computed

```python
kinematic_blend = clamp((low_thr - abs(vx)) / low_thr, 0, 1)
```

`vx` is the *forward* velocity component. The kinematic-bicycle
regularization exists to remove slip-angle singularities at parking-lot
speeds — but it is only valid when the tires are rolling, i.e. when
**total** speed is low. Keying on `vx` treated a sideways slide as a
slow vehicle. Inside the blend, `fy_f = fy_r = 0` and `alpha_z = 0`
deleted all lateral dynamics.

Secondary root cause (R2): at full saturation `Fy_i = µ·Fz_i` and the
yaw torque `a·Fz_f·cosδ − b·Fz_r ≈ a·µ·Fz_f − b·µ·Fz_r ≡ 0` by the static
load/lever relation — the model had no mechanism to re-align a released
slide. Real vehicles recover via aligning moment (pneumatic trail)
acting through the slip-dependent tire.

## 4. Complete Physics Audit

See `docs/PHYSICS_COMPLETENESS_AUDIT.md` for the full matrix. Audit
sweeps (`_tmp_view/hs_audit*.py`, steer {0.02–0.10} × v {20–44 m/s})
established:

- Peak lateral acceleration is correctly capped at ~µ·g (0.95–1.0 g at
  saturation) — the envelope is honest, not inflated.
- Tire utilization reaches 1.00 exactly at departures; sub-limit inputs
  at high speed hold stable slip states.
- Demand to saturate at 40 m/s is only ~5% of the normalized steer range
  (v² scaling of required ay) — a physically correct but narrow usable
  band; large inputs legitimately depart.

## 5. Current Physics Architecture

Planar single-track model, `dt=1/60` env step → 4 substeps at 240 Hz,
explicit Euler. States: `pos, yaw, pitch, roll, vel_body, vel_world,
yaw_rate, accel_body, accel_world, steering_angle, wheel states,
diagnostics (alpha, fy, fx, fz per axle)`.

Action pipeline: `[-1,1] steer cmd → rate-limited road wheel`;
throttle → power-limited drive force; brake → biased split;
`surface_friction` is now **per-step, per-position** (road/curb/off);
`world_accel_bias` injects banking/grade gravity.

## 6. Missing Physics (was)

- Per-point surface friction consumption (authored, never applied)
- Off-road/curb friction regions (uniform µ everywhere)
- Banking and grade effects on the chassis (authored, never applied)
- Any contact response (walls were flags)
- Tire relaxation length, load sensitivity, aligning moment
- Quasi-static body attitude for sensors/rendering

## 7. Incorrect Physics (was)

- Kinematic blend keyed on `vx` → lateral-force freeze (R1)
- Aero drag and rolling resistance acted on `vx` only — a sideways slide
  coasted aero-free
- IMU hardcoded `az=9.81` ignoring body attitude
- No static friction: a parked car on a banked road crept ~bias·h every
  substep instead of being held by the contact patch
- Tire normal loads ignored road tilt (`Fz = mg` even on a bank) —
  parked cars held on super-critical slopes that exceed tan θ = µ
- Boundary depenetration oriented the normal toward the vehicle centre —
  a car whose centre crossed the wall line was ejected OFF the track
  instead of resolved back inside
- Open-route end check had zero tolerance — a spawn sitting 1 mm behind
  the first spline sample was classified off-road and terminated
  instantly
- Config comment claimed "rear stiffer => understeer"; measured linear
  gradient actually met its intent only accidentally (+0.57°/g linear,
  +2.1°/g measured) — comment corrected with the real formula.

## 8. Incomplete Physics (was)

- `pitch`/`roll` existed but were never integrated (always 0)
- `alpha_f_eff`/`alpha_r_eff` relaxation states absent
- `CollisionResult` carried no contact geometry — response impossible
- `SplinePoint` lacked `friction`; `query_surface` did not exist

## 9. Numerical Issues

- `atan2` slip-angle denominator floored at ±0.5 m/s (kept — correct).
- Relaxation lag guarded: `max(v_abs, 0.5)·h/σ`, clamped ≤1 — cannot
  destabilize; at rest the lag decays stale slip toward zero.
- Coulomb decay inside the kinematic blend bounded by `|vy|/h` — removes
  residual vy within one step but can never reverse its sign.
- Contact impulse uses the standard rigid-body denominator
  `1/m + (r×n)²/Iz`; `e=0.15`, tangential impulse clamped to µ·Jn —
  provably non-energy-injecting.

## 10. Tire Model Assessment

`Fy = −µ·Fz·tanh(Cα·α/(µ·Fz))` per axle is a sound saturating curve:
linear slope `Cα` at origin, asymptote `µ·Fz`. After repair it acts on
the relaxation-lagged effective slip and a load-sensitive cap. It lacks
post-peak falloff (real tires lose ~15-25% beyond α*); consequence:
saturated slides are mildly *more* stable post-peak than reality —
acceptable, documented.

## 11. Combined-Slip Assessment

Demanded `(Fx, Fy)` is scaled proportionally onto the `µ·Fz` circle per
axle. Correct limit behavior verified: sustained throttle costs ~0.15 g
of usable lateral grip (earlier tests). It is a resultant-force
approximation — there is no underlying slip-ratio/wheel-speed dynamics,
so longitudinal slip transients are absent. Documented gap.

## 12. Load-Transfer Assessment

Longitudinal `ΔFz = m·ax·h/L` on proper accel, clamped ±90%. Verified
direction (braking loads the front, test
`test_loaded_axle_stiffness_sublinear`). Combined with load sensitivity,
braking gives sublinear front-grip growth — correct. Lateral transfer is
absent by construction (axle-aggregated model).

## 13. Steering Assessment

Road-wheel angle rate-limited at 4.5 rad/s; 0.58 rad limit. No
suspension/compliance feedback or self-centering torque reported to the
driver — acceptable for a torque-input action space. The v² demand
scaling at high speed is physical, not a defect; the V17 sweep maps the
usable band explicitly.

## 14. Yaw/Sideslip Assessment

- Steady-state: β settles bounded at sub-limit demands (V17 table).
- Transient: step steer delay 50% @ 0.10 s, 90% @ 0.20 s, 4.7%
  overshoot, settle 0.22 s (V06) — plausible sedan response.
- Saturated slides now self-align after input release (relaxation +
  aligning moment) and are recoverable by countersteer (0.35 s, V15).
- True >µ·g demands still depart and spin (0.10 @ 44 m/s → 90°) —
  physically required behavior preserved.

## 15. Road/Surface Physics

`query_surface(pos)` returns region (`road|curb|off`), local friction
multiplier (control-point interpolated, curb 0.85, off-road 0.55),
banking rad, grade, elevation. The environment composes it as
`µ = default_friction × scenario_mult × DR_mult × local_mult` and
supplies `world_accel_bias = −g·sin(bank)·n̂ − g·tz·t̂`. Verified: 15°
banking drifts the car 3.9 m downhill in 3 s; 6.7% grade costs 1.6 m/s
over 3 s; friction step 1.0→0.55 drops steady ay to the lower cap.

## 16. Collision/Contact Physics

Detection (SAT + spatial hash) now feeds a rigid-body contact response:
positional depenetration along the wall normal plus normal impulse
`Jn = −(1+e)·vn/(1/m + (r×n)²/Iz)` and Coulomb-capped tangential impulse,
including the yaw moment arm. V20 results: head-on 20 m/s → final
2.6 m/s, KE→1.7%, no wall crossing; oblique → KE→4.6%, no crossing.
Response is strictly dissipative (e≤1, friction opposes slip).

## 17. Sensor/Physics Consistency

The IMU reads `accel_body` — proper acceleration — which deliberately
excludes the gravity bias injected for banking/grade, and now projects
gravity through quasi-static roll/pitch (`ay+g·sin(roll)`,
`ax+g·sin(pitch)`, `az=g·cos·cos`). Verified: bias curves the trajectory
without polluting proper accel (test `test_bias_not_in_proper_accel`).

## 18. Changes Implemented

| File | Change |
|---|---|
| `sim_core/vehicle/vehicle_model.py` | blend on v_total; kinematic Coulomb vy-decay (tilt-scaled); static-friction hold at rest; per-axis aero drag; tire relaxation state; load sensitivity; pneumatic trail; `world_accel_bias`; `Fz·cosθ` tilt loads; quasi-static roll/pitch; proper-accel excludes bias |
| `sim_core/vehicle/vehicle_config.py` | 9 new physical params; corrected stiffness comments |
| `sim_core/track/spline.py` | `SplinePoint.friction` propagated + interpolated |
| `sim_core/track/road_definition.py` | `curb_friction=0.85`, `off_road_friction=0.55` (serialized); banking documented as degrees |
| `sim_core/track/mesh_generator.py` | `GeneratedTrack.road_def` retained |
| `sim_core/track/track_queries.py` | `query_surface()`; `friction`/`banking` in pose info; 0.5 m open-route end tolerance |
| `sim_core/vehicle/collision.py` | `contact_seg`/`contact_obb` on results; `apply_boundary_contact` (interior-oriented normal), `apply_obstacle_contact`, `_apply_impulse` |
| `sim_env/environment.py` | per-step surface query + friction composition + bias; contact response after detection |
| `sim_core/sensors/imu_sensor.py` | gravity through roll/pitch |
| `tools/vehicle_dynamics/harness.py` | `surface_fn`, `bias_fn`, util/mu/roll/pitch series + metrics |
| `tools/vehicle_dynamics/v_suite.py` | V01–V22 suite |
| `tests/test_vehicle_dynamics_validation.py` | +10 regression tests |

## 19. Parameters and Physical Justification

| Param | Value | Justification |
|---|---|---|
| `tire_relaxation_m` | 0.55 m | Passenger-tire relaxation lengths ~0.3–0.7 m (Pacejka) |
| `tire_load_sensitivity` | 0.10 | ~10% µ-efficiency loss per doubling of load — typical radial |
| `tire_stiffness_load_exp` | 0.8 | Cα ∝ Fz^0.8 — standard sublinear fit |
| `pneumatic_trail_m` | 0.05 | ~5 cm trail, decaying to 0 by 17° slip |
| `roll_deg_per_g` / `pitch_deg_per_g` | 4.0 / 1.2 °/g | Passenger-car roll/dive gradient range 3–7 / 1–2 °/g |
| `side_drag_coeff·side_area` | 1.0·4.9 m² | Broadside projected area, Cs≈1 |
| `contact_restitution` / `contact_friction` | 0.15 / 0.6 | Guardrail impacts are near-inelastic; tire-vs-barrier µ<1 |
| `curb_friction` / `off_road_friction` | 0.85 / 0.55 | Painted concrete curb / grass-gravel range |
| `low_speed_slide_decay` | 0.8·µg | Coulomb-bounded scrub, capped vs `|vy|/h` |

## 20. Validation Results

`benchmarks/vehicle_dynamics/v_suite/results.json` (machine-readable) +
per-maneuver CSVs. Highlights:

- V04 understeer gradient **+2.1 °/g** (mild understeer)
- V06 step steer: t50 = 0.10 s, t90 = 0.20 s, overshoot 4.7%,
  settle 0.22 s
- V15 countersteer recovery: 0.35 s, final β ≈ 0
- V17 stability grid: β grows smoothly 0.7°→4.2° across 20→44 m/s
- V18 friction transition: ay falls to the lowered cap
  (5.6→4.8 m/s², util→0.997)
- V19 banking: −4.26 m lateral drift under 10° bank
- V20 collision: KE→1.7% head-on / 4.6% oblique, no tunneling
- V21: position error vs 240 Hz shrinks 0.93→0.40→0.11→0 m
- V22: bitwise-identical deterministic runs

## 21. Standard-Referenced Test Results

| Standard | What was applied | Result |
|---|---|---|
| SAE J266 | steady-state δ vs ay sweep → understeer gradient, yaw gain | K=+2.1 °/g; yaw gain 5.36 (rad/s)/rad |
| ISO 4138 | constant-steer steady-state cornering points | all finite, monotone |
| ISO 7401 | step-steer transient metrics | delay/overshoot/settle above |
| ISO 3888-1 | double-lane-change *profile* (not the gate corridor) | peak β 7.7°, recovers to 0 — bounded |
| Pacejka | tire relaxation, load sensitivity, trail | implemented at axle level |
| Chrono | impulse contact with friction | implemented at rigid-body level |

## 22. Before/After Comparison

| Scenario | Before | After |
|---|---|---|
| 40 m/s slide → vx<1.5 while sliding | **vy≈18 m/s for 15+ s, −175 m drift, yaw frozen** | vy→0 in ~3 s, bounded slide |
| 40 m/s saturated slide, hands-off | persistent spin/slide | **self-recovers β→0 in ~0.8 s** |
| 40 m/s slide + countersteer | recovered (0.5 s) | recovered (0.35 s) |
| 0.08 steer @ 40 m/s transient | β 39.8° departure | β 5.7° bounded (relaxation damps overshoot) |
| 0.10 steer @ 44 m/s | spin (physical) | spin 89.7° — preserved |
| Authored low-µ region | no effect | grip drops, car slides wider |
| Banking | ignored | lateral gravity pulls car downhill |
| Parked on 15° bank | crept −0.011 m/s continuously | holds at 0.00 (static friction); 50° bank (>tan⁻¹µ) slides to the wall |
| Barrier impact | flag only; vehicle drove through | impulse stops/deflects car, KE→~2-5%, interior-oriented depenetration |
| Spawn at open-route start | off-road → instant termination | 0.5 m end tolerance → on-road |

## 23. Remaining Limitations

- No longitudinal slip-ratio/wheel-speed dynamics — drivetrain Fx is
  demand + envelope (documented approximation).
- No post-peak tire falloff — saturated slides are marginally too stable.
- No lateral load transfer (single-track abstraction).
- Contact response allows a transient ≤~0.3 m overshoot beyond the wall
  line between steps at 20 m/s (resolved the same step; not a tunnel).
- Grazing wall contact sustains almost no normal force, so wall-scrape
  friction is under-damped relative to reality (rigid-impulse limitation).
- `open`/`invisible` boundary types still produce collision segments.
- No reverse gear; `reverse_max_speed` declared unused.
- Pre-existing test breakage in `tests/agent/*` (missing `smoke` track in
  registry) is unrelated to physics — 12 failures reproduced on clean HEAD.

## 24. Evidence Index

- `benchmarks/vehicle_dynamics/v_suite/results.json` + `csv/` — V01–V22
- `benchmarks/vehicle_dynamics/repaired/` — standard maneuver bench + plots
- `_tmp_view/vy_freeze.py`, `hs_audit2.py`, `hs_recovery*.py`,
  `env_physics_check.py` — reproduction/verification scripts
- `tests/test_vehicle_dynamics_validation.py` — 30 model/validation tests
- `docs/PHYSICS_COMPLETENESS_AUDIT.md` — full subsystem matrix

## 25. References

- Pacejka, *Tire and Vehicle Dynamics* — slip angles, relaxation length,
  load sensitivity, pneumatic trail, friction ellipse.
- SAE J266 — steady-state directional control (understeer gradient
  methodology only).
- ISO 4138 — steady-state circular driving; ISO 7401 — step-steer
  transient metrics (delay/overshoot/settling); ISO 3888-1 — double lane
  change (profile inspiration only).
- Project Chrono vehicle models — rigid-body impulse + friction contact
  formulation reference.
