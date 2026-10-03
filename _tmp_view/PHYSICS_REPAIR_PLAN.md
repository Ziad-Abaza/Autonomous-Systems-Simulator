---
noteId: "ddfeb000bf7911f1a29f1fbaabbd87c8"
tags: []

---

# Vehicle & Environment Physics Realism Repair — Implementation Plan

**Goal:** Repair vehicle/environment physics so high-speed behavior is physically coherent — fix the confirmed lateral-force freeze, wire authored surface friction/banking into dynamics, add real tire transient/load effects, add collision response — and prove it with a V01–V22 standards-aligned validation suite and two audit documents.

**Architecture:** Keep the single-track (bicycle) dynamic model — legitimate documented abstraction (Pacejka Ch. 1). All changes are physically motivated additions plus environment wiring; no friction inflation, no damping cheats, no trajectory forcing.

## Root Causes (Phase 1 findings)

| # | Finding | Evidence |
|---|---------|----------|
| R1 | Kinematic blend keyed on `|vx|` kills ALL lateral force when a slide scrubs vx below ~1.5 m/s → car glides sideways ~indefinitely | `vy_freeze.py`: vy 20.9→18.1 over 15 s, −175 m lateral drift |
| R2 | Hands-off saturated slide has ~0 self-aligning yaw torque (a·Fz_f ≡ b·Fz_r; monotonic tanh, no aligning moment). Correct countersteer DOES recover (β→0 in 0.5 s @40) | `hs_recovery2.py` |
| R3 | `ControlPoint.friction` dropped by spline builder; `banking` stored but unused; no off-road/curb friction; collision detection-only; drag on vx only; pitch/roll dead; `reverse_max_speed` dead config | code review |

## Tasks

1. **Blend fix**: key on `v_total`; bounded Coulomb lateral decay in kinematic regime.
2. **Tire upgrades**: drag on velocity vector; relaxation length σ=0.55 m; load sensitivity (cap + stiffness); pneumatic trail 0.05 m decaying to 0 by |α|=0.3 rad.
3. **Roll/pitch quasi-static** + `world_accel_bias` input; IMU gravity projection consistent.
4. **Track surface physics**: friction into SplinePoint; `query_surface()` → region friction/banking/grade; env per-step surface + bias.
5. **Collision response**: depenetration + inelastic impulse + Coulomb friction + yaw arm. Never adds KE.
6. **V01–V22 suite**: `tools/vehicle_dynamics/v_suite.py` + extended maneuvers + metrics.
7. **Docs**: PHYSICS_COMPLETENESS_AUDIT.md, PHYSICS_REALISM_REPAIR_REPORT.md, vehicle-model.md update.
8. **Regression pass**: full pytest green.
