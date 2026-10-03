---
noteId: "0b8d7dd0bf7f11f1a29f1fbaabbd87c8"
tags: []

---

# Physics Completeness Audit

Complete inventory of the vehicle/environment physics model after the
physics-realism repair. For every subsystem: what is modeled, the
mathematical formulation used, what is approximated, what is missing, and
whether the approximation produces correct physical behavior at the limit.

Legend: **M** = implemented model · **A** = approximation (defensible
simplification) · **N** = not modeled (documented gap).

---

## Vehicle Dynamics Core

| Subsystem | Status | Implementation | Notes / Limitations |
|---|---|---|---|
| Chassis model | M | Single-track (bicycle) dynamic, planar `vx/vy/wz` | Legitimate research abstraction (Pacejka Ch.1). No heave/pitch/roll DOF. |
| Integration | M | Explicit Euler, `physics_substeps=4` → 240 Hz inside 60 Hz steps | V21: position deltas vs 240 Hz converge monotonically (0.93 m @30 Hz). |
| Steering geometry | A | Rate-limited road-wheel angle; no kinematic Ackermann/toe | `steering_rate=4.5 rad/s` rate limit is physical; no steering compliance. |
| Longitudinal drive | A | Power-limited force demand `F=min(Fmax, P/v)` | No gear ratio, wheel-speed or slip-ratio dynamics — see "Not modeled". |
| Braking | M | Total force × `brake_bias_front` split, opposes vx | No ABS/wheel lock model; envelope clamp gives lockup-like saturation. |
| Aerodynamics | A | Per-axis quadratic drag: frontal `CxA` on vx, side `CsAs` on vy | Lateral aero damps slides realistically; no lift/downforce or aero yaw moment. |
| Rolling resistance | M | `Crr·m·g` opposing vx | Not applied to lateral sliding (that resistance is carried by tire Fy). |
| Longitudinal load transfer | M | `ΔFz = m·ax·h/L`, clamped ±90% | Uses proper acceleration — correct quantity. |
| Lateral load transfer | N | — | Single-track axle aggregation makes it unobservable at axle level; it would only matter for inside/outside wheel saturation split. |
| Quasi-static attitude | A | `roll = 4°/g·ay`, `pitch = 1.2°/g·ax` | Feeds IMU gravity projection and camera pose; no roll/pitch dynamics or damping. |

## Tire Model

| Subsystem | Status | Implementation | Notes / Limitations |
|---|---|---|---|
| Tire curve shape | A | `Fy = −µ·Fz·tanh(Cα·α/(µ·Fz))` | Correct initial slope `Cα`, asymptotes to `µ·Fz`. No post-peak falloff — a real tire loses ~15-25% peak force at large slip; documented. |
| Slip-angle definition | M | `α = atan2(vy ± a·wz, vx_eff)` signed, `vx_eff` floor ±0.5 m/s | Correct at all headings including deep slides and reversing. |
| Cornering stiffness | M | `Cα_f = 80000`, `Cα_r = 85000` N/rad → ~13-15/rad normalized | Understeer gradient +2.1 °/g measured (V04). |
| Combined slip | A | Demanded `(Fx, Fy)` scaled proportionally into `µ·Fz` circle per axle | Captures the dominant effect (power eats lateral grip); not a full Pacejka combined-slip — no slip-ratio model underneath Fx. |
| Tire relaxation | M | First-order lag on effective slip, `σ = 0.55 m` | Physical contact-patch transient; produces finite yaw-response delay (V06: 50% @ 0.10 s). |
| Load sensitivity | M | cap `µFz/(1+k·(Fz/Fz₀−1))`, k=0.10; stiffness ∝ `Fz^0.8` | Braking gains front grip sublinearly — correct direction and magnitude. |
| Aligning moment | M | `Mz = −t·Fy`, `t = 0.05 m·max(0, 1−|α|/0.3)` | Restores hands-off slide self-alignment (R2 fix). |
| Tire temp/wear/pressure | N | — | Not modeled. |

## Low-Speed / Singularity Handling

| Subsystem | Status | Implementation | Notes / Limitations |
|---|---|---|---|
| Kinematic blend | M | Blend on **total speed** `hypot(vx,vy) < 1.5 m/s` | R1 fix: previously keyed on `|vx|` — a sideways slide killed all lateral force (the freeze bug). |
| Residual slide decay | M | Coulomb-bounded lateral decay `min(0.8·µg·cosθ, |vy|/h)` inside blend | Slow slides stop instead of gliding; tilt-scaled so super-critical slopes still slide. |
| Static friction hold | M | At rest, gravity bias ≤ `µg·cosθ` is cancelled by the patch reaction | A parked car holds on a 15° bank; releases and slides at tan θ > µ. |
| Kinematic yaw | M | `wz → vx·tanδ/L` via bounded relaxation τ=0.05 s | No deadbeat forcing. |

## Environment / Road Physics

| Subsystem | Status | Implementation | Notes / Limitations |
|---|---|---|---|
| Per-point surface friction | M | `ControlPoint.friction` → spline interpolation → `query_surface` | Previously authored but silently dropped (R3a fix). |
| Off-road / curb friction | M | `road_def.off_road_friction=0.55`, `curb_friction=0.85` | Region classification by lateral offset vs width/curb band. |
| Banking physics | M | Authored degrees → `g·sin(bank)` world bias toward downhill; normal loads scale `cos θ` | Previously unused (R3b fix). |
| Grade physics | M | `−g·tangent.z` along horizontal tangent | Hills decelerate the car (verified −1.6 m/s over 3 s climb). |
| Elevation | A | `pos.z` pinned to surface elevation | No vertical dynamics/suspension travel. |
| Surface bias ↔ proper accel | M | `world_accel_bias` integrates velocity but excluded from `accel_body` | IMU correctly does not feel gravity. |

## Collision / Contact

| Subsystem | Status | Implementation | Notes / Limitations |
|---|---|---|---|
| Boundary detection | M | OBB-vs-segment SAT + spatial-hash broadphase | |
| Boundary response | M | Depenetration + rigid impulse `Jn=−(1+e)vn/m_eff`, Coulomb tangential, yaw arm; normal oriented toward road interior | e=0.15, µc=0.6 — strictly removes energy (V20: KE→1.7%). Interior-oriented normal prevents wrong-side ejection. Grazing contact underestimates sustained wall friction (impulse only fires while approaching) — documented. |
| Obstacle contact | M | Closest-point normal + same impulse law | |
| Vehicle–vehicle | N | — | Not modeled. |
| Boundary typing | A | Collision segments exist for every boundary type incl. `open`/`invisible` | Authored "open" edges still behave as walls; flag-driven filtering is a documented semantic gap, not a physics defect. |

## Sensors ↔ Physics Consistency

| Subsystem | Status | Implementation | Notes / Limitations |
|---|---|---|---|
| IMU accel | M | `accel_body` (proper accel) + gravity through roll/pitch projection | Consistent with planar model; correct on banking (reads only tire/aero force). |
| IMU gyro | A | yaw rate only | No roll/pitch rate channels (planar model has none to measure). |
| Vehicle-state sensor | M | reads state directly | Consistent by construction. |
| Wheel speeds | A | `vx/r` free-roll estimate | Cosmetic for visuals; no slip-ratio wheels exist. |

## Explicitly Not Modeled (documented gaps)

- Wheel rotational dynamics / longitudinal slip ratio (drivetrain Fx is demand+envelope, not slip-generated).
- Lateral load transfer (single-track abstraction).
- Reverse gear (no gear input; `reverse_max_speed` is dead config — declared unused).
- Suspension kinematics, dampers, ride height.
- Tire post-peak force falloff, temperature, wear.
- Aerodynamic lift/downforce and yaw moment.
- Vertical dynamics (`pos.z` follows surface elevation).
- Vehicle–vehicle contact.

## Summary

The physics **defects** — the lateral-force freeze in the low-speed blend,
authored surface friction/banking never reaching the tires, and
detection-only collision — are repaired and regression-tested. The
remaining items are deliberate single-track simplifications: the model is
a research-grade planar bicycle with Pacejka-style tire effects
(relaxation, load sensitivity, aligning moment), at the fidelity tier
commonly used in autonomous-driving/RL literature.
