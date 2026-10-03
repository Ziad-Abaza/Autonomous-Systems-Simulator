"""
Modular vehicle physics package.
"""

from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.vehicle.vehicle_model import VehicleModel, VehicleState
from sim_core.vehicle.collision import VehicleCollisionChecker, CollisionResult

__all__ = [
    'VehicleConfig',
    'VehicleModel',
    'VehicleState',
    'VehicleCollisionChecker',
    'CollisionResult',
]
