"""
Vehicle configuration parameters for physical modeling and modular chassis customization.

Target vehicle class: generic passenger vehicle. All values are representative
assumptions, not measurements from a specific real vehicle.
Units: SI (kg, m, s, N, rad) unless noted.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class VehicleConfig:
    # Mass and geometry
    mass: float = 1200.0             # kg
    length: float = 4.2              # m
    width: float = 1.8               # m
    height: float = 1.4              # m
    wheelbase: float = 2.6           # distance between front and rear axles (m)
    track_width: float = 1.5         # width between left and right wheels (m)
    cog_height: float = 0.45         # center of gravity height (m); used for longitudinal load transfer
    weight_dist_front: float = 0.52  # fraction of mass over front axle [0-1]
    wheel_radius: float = 0.34       # m
    yaw_inertia: float = 0.0         # kg.m^2; 0 = auto-derived as m*a*b (a,b = CG-to-axle)

    # Steering limits and response
    max_steering_angle: float = 0.58 # rad (~33 deg) road-wheel limit
    steering_rate: float = 4.5       # rad/s max road-wheel rate

    # Powertrain & Brakes
    # max_drive_force is the low-speed tractive bound (gearing/edge-of-
    # traction limit), NOT the engine torque demand: sized ~0.88x the
    # rear axle's static friction cap (~5650 N) because a RWD with no
    # TC/differential model cannot transmit more without wheelspin.
    # At speed the engine is power-limited: F = engine_power / v.
    max_drive_force: float = 5000.0  # N at tire contact patches
    engine_power: float = 90000.0    # W (~120 hp, generic compact car)
    power_min_speed: float = 2.0     # m/s, floor for the P/v term at crawl
    max_brake_force: float = 9000.0  # N total across axles
    top_speed: float = 45.0          # m/s (~162 km/h), speed limiter band
    reverse_max_speed: float = 10.0  # m/s
    drive_force_front_fraction: float = 0.0   # 0 = RWD, 1 = FWD, 0.5 = symmetric AWD
    brake_bias_front: float = 0.65            # fraction of brake force on front axle

    # Resistance and Friction
    drag_coeff: float = 0.32         # aerodynamic drag Cd (frontal)
    frontal_area: float = 2.1        # m^2
    side_drag_coeff: float = 1.0     # aero side-force coefficient (broadside)
    side_area: float = 4.9           # m^2 projected side area (L*H*0.83)
    air_density: float = 1.225       # kg/m^3
    rolling_resistance: float = 0.015  # dimensionless
    tire_friction: float = 1.0       # tire-road mu multiplier

    # Tire cornering stiffness (per-axle aggregate, N/rad of slip angle).
    # Values chosen so normalized stiffness Ca/(mu*Fz) ~= 13-15 /rad,
    # representative of passenger tires (baseline audit found 14000/16000,
    # i.e. ~2.5/rad, which produced 6-8x too much sideslip).
    # Steady-state balance: understeer gradient
    #   K = (Wf/CaF - Wr/CaR)/g > 0 <=> understeer.
    # With wdf=0.52 (Wf=6120 N, Wr=5650 N), CaF=80000/CaR=85000 gives
    # Wf/CaF - Wr/CaR > 0 -> mild understeer (measured ~2 deg/g including
    # nonlinear trail/saturation effects; see V04 in the validation suite).
    cornering_stiffness_front: float = 80000.0  # N/rad
    cornering_stiffness_rear: float = 85000.0   # N/rad (front/axle-load ratio keeps K>0)

    # Tire model selection.
    #   "bicycle"   — axle-aggregate saturating-tanh (default; validated).
    #   "pacejka4"  — per-wheel magic-formula Pacejka with 4-corner normal
    #                 loads, longitudinal+lateral load transfer, post-peak
    #                 force decay, and per-wheel friction-envelope coupling.
    tire_model: str = "bicycle"
    pacejka_shape_c: float = 1.3          # magic-formula shape factor (lateral)
    roll_stiffness_front_fraction: float = 0.55  # lateral transfer split (front share)

    # Tire transient & load effects (Pacejka-style, axle-aggregate)
    tire_relaxation_m: float = 0.55      # relaxation length (m); force lags slip by ~sigma/v seconds
    tire_load_sensitivity: float = 0.10  # cap efficiency loss: cap = mu*Fz/(1+k*(Fz/Fz_static-1))
    tire_stiffness_load_exp: float = 0.8 # C_alpha scales ~ Fz^exp (sublinear load sensitivity)
    pneumatic_trail_m: float = 0.05      # aligning-moment lever arm; decays to 0 at |alpha|=0.3 rad

    # Quasi-static suspension attitude (fed to IMU/cameras; no roll dynamics)
    roll_deg_per_g: float = 4.0          # chassis roll per unit lateral g (right-side-down +)
    pitch_deg_per_g: float = 1.2         # chassis pitch per unit longitudinal g (nose-up +)

    # Contact response (rigid impulse vs boundaries/obstacles)
    contact_restitution: float = 0.15    # normal-velocity restitution e (0 = inelastic)
    contact_friction: float = 0.6        # tangential Coulomb friction vs walls

    # Numerics
    physics_substeps: int = 4        # internal substeps per step(); 60Hz env -> 240Hz physics
    low_speed_threshold: float = 1.5 # m/s TOTAL speed below which kinematic blend engages
    low_speed_relax_tau: float = 0.05  # s yaw-rate relaxation time inside blend
    low_speed_slide_decay: float = 0.8 # fraction of mu*g used to kill residual lateral slide in kinematic regime

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleConfig:
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, type(getattr(config, k))(v))
        return config
