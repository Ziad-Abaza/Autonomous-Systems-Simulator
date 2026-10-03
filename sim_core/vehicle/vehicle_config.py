"""
Vehicle configuration parameters for physical modeling and modular chassis customization.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class VehicleConfig:
    # Mass and geometry
    mass: float = 1200.0             # kg
    length: float = 4.2              # meters
    width: float = 1.8               # meters
    height: float = 1.4              # meters
    wheelbase: float = 2.6           # distance between front and rear axles (m)
    track_width: float = 1.5         # width between left and right wheels (m)
    cog_height: float = 0.45         # center of gravity height (m)
    weight_dist_front: float = 0.52  # fraction of mass over front axle

    # Steering limits and response
    max_steering_angle: float = 0.58 # ~33 degrees (rad)
    steering_rate: float = 4.5       # rad/s max rate of turn

    # Powertrain & Brakes
    max_drive_force: float = 6500.0  # Newtons
    max_brake_force: float = 9000.0  # Newtons
    top_speed: float = 45.0          # m/s (~162 km/h)
    reverse_max_speed: float = 10.0  # m/s

    # Resistance and Friction
    drag_coeff: float = 0.32         # aerodynamic drag Cd
    frontal_area: float = 2.1        # m^2
    air_density: float = 1.225       # kg/m^3
    rolling_resistance: float = 0.015
    tire_friction: float = 1.0       # baseline friction multiplier

    # Cornering stiffness
    cornering_stiffness_front: float = 14000.0  # N/rad
    cornering_stiffness_rear: float = 16000.0   # N/rad

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleConfig:
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, float(v))
        return config
