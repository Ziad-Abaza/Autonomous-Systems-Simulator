"""
Scenario definitions separating environmental conditions, weather,
lighting, and obstacles from physical road geometry.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from sim_core.world.obstacle import Obstacle


@dataclass
class ScenarioConfig:
    name: str = "Standard Day"
    time_of_day: str = "day"        # "day", "dusk", "night"
    weather: str = "clear"          # "clear", "rain", "fog"
    ambient_light: float = 1.0      # 0.1 (dark night) to 1.0 (bright day)
    surface_friction_mult: float = 1.0  # e.g., 0.65 for wet rain
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    difficulty_level: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScenarioConfig:
        return cls(
            name=str(data.get('name', 'Standard Day')),
            time_of_day=str(data.get('time_of_day', 'day')),
            weather=str(data.get('weather', 'clear')),
            ambient_light=float(data.get('ambient_light', 1.0)),
            surface_friction_mult=float(data.get('surface_friction_mult', 1.0)),
            obstacles=list(data.get('obstacles', [])),
            difficulty_level=int(data.get('difficulty_level', 1)),
        )

    @classmethod
    def get_presets(cls) -> Dict[str, ScenarioConfig]:
        """Built-in scenario presets."""
        return {
            "day_clear": cls(name="Day Clear", time_of_day="day", weather="clear", ambient_light=1.0, surface_friction_mult=1.0),
            "wet_rain": cls(name="Wet Rain", time_of_day="dusk", weather="rain", ambient_light=0.6, surface_friction_mult=0.7),
            "night_fog": cls(name="Night Fog", time_of_day="night", weather="fog", ambient_light=0.25, surface_friction_mult=0.9),
        }
