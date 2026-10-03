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
    max_drive_force: float = 6500.0  # N at tire contact patches
    max_brake_force: float = 9000.0  # N total across axles
    top_speed: float = 45.0          # m/s (~162 km/h), linear power taper
    reverse_max_speed: float = 10.0  # m/s
    drive_force_front_fraction: float = 0.0   # 0 = RWD, 1 = FWD, 0.5 = symmetric AWD
    brake_bias_front: float = 0.65            # fraction of brake force on front axle

    # Resistance and Friction
    drag_coeff: float = 0.32         # aerodynamic drag Cd
    frontal_area: float = 2.1        # m^2
    air_density: float = 1.225       # kg/m^3
    rolling_resistance: float = 0.015  # dimensionless
    tire_friction: float = 1.0       # tire-road mu multiplier

    # Tire cornering stiffness (per-axle aggregate, N/rad of slip angle).
    # Values chosen so normalized stiffness Ca/(mu*Fz) ~= 13-15 /rad,
    # representative of passenger tires (baseline audit found 14000/16000,
    # i.e. ~2.5/rad, which produced 6-8x too much sideslip).
    cornering_stiffness_front: float = 80000.0  # N/rad
    cornering_stiffness_rear: float = 85000.0   # N/rad (rear > front => mild understeer)

    # Numerics
    physics_substeps: int = 4        # internal substeps per step(); 60Hz env -> 240Hz physics
    low_speed_threshold: float = 1.5 # m/s below which kinematic blend engages
    low_speed_relax_tau: float = 0.05  # s yaw-rate relaxation time inside blend

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleConfig:
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, type(getattr(config, k))(v))
        return config
