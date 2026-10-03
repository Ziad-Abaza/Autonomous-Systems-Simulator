"""
Core simulation package.
Decoupled physics, track geometry, sensors, and world state.
"""

from sim_core.clock import FixedClock
from sim_core.math_utils import Vec2, Vec3, OBB2D

__all__ = [
    'FixedClock',
    'Vec2',
    'Vec3',
    'OBB2D',
]
