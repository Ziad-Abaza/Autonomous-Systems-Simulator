import os, io
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def patch(path, old, new, count=1):
    p = os.path.join(ROOT, path)
    s = io.open(p, encoding='utf-8').read()
    if old not in s:
        print(f'MISS in {path}: {old[:60]!r}')
        return
    s = s.replace(old, new, count)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
    print(f'patched {path}')

def append(path, text):
    p = os.path.join(ROOT, path)
    io.open(p, 'a', encoding='utf-8', newline='\n').write(text)
    print(f'appended {path}')

# ---------- 1. VEHICLE_DYNAMICS_VALIDATION_REPORT.md — post-repair addendum ----------
append('docs/VEHICLE_DYNAMICS_VALIDATION_REPORT.md', """

---

# Post-repair update (2026-10-04)

A control-degradation regression was found *in this report's own changes*:
the demand-first friction ellipse (`u = Fx_demand/(mu*Fz)`, `lat_cap =
mu*Fz*sqrt(1-u^2)`) treats demanded longitudinal force as delivered, so
under power the driven axle's lateral capacity collapses — with RWD +
6500 N demand vs ~5650 N rear cap, `u -> 1` below ~25 m/s and any steering
input under throttle produced an unrecoverable power-oversteer spin
(measured: launch + steer 0.10 -> beta > 60 deg, wz -88 dps).

Fixes (full analysis in `VEHICLE_DYNAMICS_REPAIR_REPORT.md`):

- Friction envelope -> proportional resultant clamp: demanded (Fx, Fy)
  scaled by `min(1, mu*Fz/|F|)`; envelope invariant preserved.
- Drive force -> power-limited `min(F_max, engine_power/|v|)` + 2 m/s
  limiter band at top_speed (replaces linear taper).
- max_drive_force 6500 -> 5000 N (traction-feasible for the RWD axle);
  engine_power = 90 kW, power_min_speed = 2.0 added.

Re-validated (tag `repaired`, 17 maneuvers): full-throttle small-steer
correction peak beta 3.18 deg (was spin); throttle-in-corner slides are
bounded and recover on lift; provoked drift 26-68 deg recovers hands-off;
timestep independence <0.7 m / <0.7 deg at 30-240 Hz; determinism bitwise;
PD baseline eval return improved on all tracks with 0 collisions.
""")

# ---------- 2. VEHICLE_DYNAMICS_MODEL.md — equations ----------
patch('docs/VEHICLE_DYNAMICS_MODEL.md',
"""    F_drive      = throttle * F_max * max(0, 1 - vx/top_speed)""",
"""    F_drive      = throttle * min(F_max, engine_power / max(power_min_speed, |vx|))
                   (power-limited engine; smooth 2 m/s limiter band at top_speed)""")

patch('docs/VEHICLE_DYNAMICS_MODEL.md',
"""    u_i   = clamp(Fx_i / (mu*Fz_i), -1, 1);  Fx_i <- u_i*mu*Fz_i     (Fx cap)
    Fy_i  = -mu*Fz_i*sqrt(1-u_i^2) * tanh(C_i*alpha_i/(mu*Fz_i))     (ellipse)""",
"""    Fy_i  = -mu*Fz_i * tanh(C_i*alpha_i/(mu*Fz_i))                 (free-rolling demand)
    s_i   = min(1, mu*Fz_i / |(Fx_i, Fy_i)|);  (Fx_i, Fy_i) <- s_i * (Fx_i, Fy_i)
              (friction envelope: demanded force vector scaled proportionally
               to fit mu*Fz — excess demand becomes slide, not lost lateral grip)""")

patch('docs/VEHICLE_DYNAMICS_MODEL.md',
"""mass 1200 kg, wheelbase 2.6 m, track 1.5 m, h_cg 0.45 m,
weight_dist_front 0.52, wheel_radius 0.34 m, Iz = m*a*b ~ 2024 kg.m2
(overridable via yaw_inertia), max_steer 0.58 rad, steer rate 4.5 rad/s,
F_drive 6500 N, F_brake 9000 N, top_speed 45 m/s, Cd 0.32, A 2.1 m2,
rho 1.225, c_rr 0.015, mu 1.0, Cf 80000 N/rad, Cr 85000 N/rad
(normalized ~13-15/rad), RWD drive split, 65% front brake bias.""",
"""mass 1200 kg, wheelbase 2.6 m, track 1.5 m, h_cg 0.45 m,
weight_dist_front 0.52, wheel_radius 0.34 m, Iz = m*a*b ~ 2024 kg.m2
(overridable via yaw_inertia), max_steer 0.58 rad, steer rate 4.5 rad/s,
F_drive_max 5000 N (traction-informed tractive bound), engine_power 90 kW,
F_brake 9000 N, top_speed 45 m/s (limiter band), Cd 0.32, A 2.1 m2,
rho 1.225, c_rr 0.015, mu 1.0, Cf 80000 N/rad, Cr 85000 N/rad
(normalized ~13-15/rad), RWD drive split, 65% front brake bias.""")

# ---------- 3. docs/vehicle-dynamics/vehicle-model.md ----------
patch('docs/vehicle-dynamics/vehicle-model.md',
"""### Tire model — friction ellipse + saturation

Each axle's lateral capacity is what's left after the longitudinal force
consumes grip:

```
u        = Fx_clamped / (µ · Fz)
lat_cap  = µ · Fz · sqrt(1 - u²)
Fy       = -lat_cap · tanh(Cα · α / max(1, cap))
```

So hard acceleration or braking *reduces* available cornering force —
combined-slip behavior without a full Pacejka implementation. Rear
cornering stiffness exceeds front by default → mild understeer.""",
"""### Tire model — friction envelope + saturation

Each axle computes a free-rolling lateral demand, then the demanded
(Fx, Fy) force vector is scaled proportionally to fit the friction
circle:

```
Fy_demand = -µ·Fz · tanh(Cα · α / max(1, µ·Fz))
s         = min(1, µ·Fz / |(Fx, Fy_demand)|)
(Fx, Fy) ← s · (Fx, Fy_demand)
```

So hard acceleration or braking *reduces* available cornering force —
combined-slip behavior without a full Pacejka implementation — but the
clamp is applied to the whole demanded vector rather than giving Fx
first claim (the earlier demand-first ellipse collapsed lateral grip to
zero under full throttle and caused power-oversteer spins). Rear
cornering stiffness exceeds front by default → mild understeer.""")

patch('docs/vehicle-dynamics/vehicle-model.md',
"""- Drive force tapers linearly to zero at `top_speed` (45 m/s ≈ 162 km/h).""",
"""- Engine force is power-limited: `F = min(max_drive_force,
  engine_power / |v|)` — tractive bound at crawl, `P/v` at speed — with a
  smooth 2 m/s limiter band at `top_speed` (45 m/s ≈ 162 km/h).""")

patch('docs/vehicle-dynamics/vehicle-model.md',
"""| `max_drive_force` | float | 6500.0 | N | Tire drive force at contact |""",
"""| `max_drive_force` | float | 5000.0 | N | Low-speed tractive bound (~traction limit of RWD axle) |
| `engine_power` | float | 90000.0 | W | Engine power — `F = P/v` at speed |
| `power_min_speed` | float | 2.0 | m/s | Floor for the P/v term |""")

patch('docs/vehicle-dynamics/vehicle-model.md',
"""| `top_speed` | float | 45.0 | m/s | Linear power taper endpoint |""",
"""| `top_speed` | float | 45.0 | m/s | Speed-limiter band edge |""")

# ---------- 4. MASTER_REQUIREMENTS_TRACEABILITY.md ----------
patch('docs/MASTER_REQUIREMENTS_TRACEABILITY.md',
"""| VD-002 | VD | Friction ellipse | COMPLETE | `vehicle_model.py` u-clamp + sqrt(1-u²) cap; asserted in tests | A/B | model | ellipse asserts | — | — | low |""",
"""| VD-002 | VD | Friction envelope | COMPLETE | `vehicle_model.py` proportional resultant clamp `min(1, µFz/|F|)` on (Fx,Fy); envelope asserted in tests; **revised 2026-10-04** — demand-first ellipse caused power-oversteer spins (see VEHICLE_DYNAMICS_REPAIR_REPORT.md) | A | model | ellipse asserts + TestCombinedSlipAllocation | `benchmarks/vehicle_dynamics/repaired` | — | low |""")

patch('docs/MASTER_REQUIREMENTS_TRACEABILITY.md',
"""| VD-010 | VD | Maneuver harness + metrics | COMPLETE | `tools/vehicle_dynamics/`; this audit executed full suite fresh | A | harness | 13 VD tests | evidence/vehicle/audit_run | >30 m/s, mu<0.6 unvalidated; no post-peak; detection-only collision | medium |""",
"""| VD-010 | VD | Maneuver harness + metrics | COMPLETE | `tools/vehicle_dynamics/`; this audit executed full suite fresh; suite extended to 17 maneuvers (s-curve, throttle-in-corner, drift init/recovery, full-throttle correction) | A | harness | 16 VD tests | evidence/vehicle/audit_run + benchmarks/vehicle_dynamics/repaired | >30 m/s, mu<0.6 unvalidated; no post-peak; detection-only collision | medium |
| VD-011 | VD | Powertrain traction feasibility + envelope allocation | COMPLETE | `max_drive_force` 6500→5000 N (≈0.88× rear static cap); power-limited engine `min(F_max, P/v)` + limiter band; proportional envelope clamp; full-throttle steer correction β 3.18° (was spin >60°) | A | model+config | TestCombinedSlipAllocation (3 new) | suite tag `repaired`; dt-check <0.7 m/0.7°; PD baseline eval improved | — | low |""")

# ---------- 5. MASTER_PROJECT_STATUS.md ----------
patch('docs/MASTER_PROJECT_STATUS.md',
"""| VD — Dynamics | 10 | 10 | 0 | 0 | 0 | Validated within stated envelope |""",
"""| VD — Dynamics | 11 | 11 | 0 | 0 | 0 | Validated within stated envelope; envelope-allocation + powertrain repaired 2026-10-04 |""")

patch('docs/MASTER_PROJECT_STATUS.md',
"""| **Total audited** | **136 (+5 superseded)** |""",
"""| **Total audited** | **137 (+5 superseded)** |""")

patch('docs/MASTER_PROJECT_STATUS.md',
"""| COMPLETE | 103 |""",
"""| COMPLETE | 104 |""")

# ---------- 6. MASTER_GAP_ANALYSIS.md — append VD gap entry ----------
append('docs/MASTER_GAP_ANALYSIS.md', """
## GAP-15 — Vehicle-dynamics envelope allocation (VD-002/VD-011) — CLOSED 2026-10-04

- **Found:** post-`7481f50` the demand-first friction ellipse + untransmissible
  RWD drive force (6500 N vs ~5650 N rear cap) collapsed rear lateral capacity
  under throttle — full-throttle + small steer input spun the car (β > 60°).
- **Fixed:** proportional resultant clamp on (Fx, Fy); power-limited engine
  `min(F_max, P/v)`; `max_drive_force` → 5000 N. Evidence:
  `benchmarks/vehicle_dynamics/repaired`, `VEHICLE_DYNAMICS_REPAIR_REPORT.md`,
  `TestCombinedSlipAllocation`.
- **Residual limits (documented, not blockers):** sustained full-throttle
  mid-corner holds a recoverable drift (physical RWD); hard brake+steer spins
  (no ABS); lane-change-grade demands depart (no ESC); axle-aggregate tires,
  no post-peak falloff; µ<0.5 / >35 m/s unvalidated.
""")
