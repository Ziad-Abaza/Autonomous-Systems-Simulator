"""
Scenario Designer.
Enables creating reusable training and evaluation scenarios referencing
the base environment configuration with clean parametric overrides.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from sim_env.randomization_designer import DomainRandomizationDefinition


@dataclass
class ScenarioDefinition:
    """
    Declarative training scenario representing a specific environmental situation.
    References the base environment and applies parametric overrides.
    """
    scenario_id: str
    name: str
    description: str = ""
    weather: str = "clear"               # "clear", "rain", "fog"
    time_of_day: str = "day"             # "day", "dusk", "night"
    ambient_light: float = 1.0           # 0.1 (dark night) to 1.0 (bright day)
    surface_friction_mult: float = 1.0   # Multiplier on road surface friction
    target_speed_override: Optional[float] = None
    time_limit_override: Optional[float] = None
    sensor_noise_mult: float = 1.0
    spawn_override: Optional[Dict[str, Any]] = None  # Override spawn position or yaw
    obstacle_overrides: List[Dict[str, Any]] = field(default_factory=list)
    randomization: DomainRandomizationDefinition = field(default_factory=DomainRandomizationDefinition.create_default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "weather": self.weather,
            "time_of_day": self.time_of_day,
            "ambient_light": self.ambient_light,
            "surface_friction_mult": self.surface_friction_mult,
            "target_speed_override": self.target_speed_override,
            "time_limit_override": self.time_limit_override,
            "sensor_noise_mult": self.sensor_noise_mult,
            "spawn_override": self.spawn_override,
            "obstacle_overrides": self.obstacle_overrides,
            "randomization": self.randomization.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScenarioDefinition:
        rand_data = data.get("randomization", {})
        rand_def = DomainRandomizationDefinition.from_dict(rand_data) if rand_data else DomainRandomizationDefinition.create_default()

        return cls(
            scenario_id=str(data.get("scenario_id", "default_scenario")),
            name=str(data.get("name", "Default Scenario")),
            description=str(data.get("description", "")),
            weather=str(data.get("weather", "clear")),
            time_of_day=str(data.get("time_of_day", "day")),
            ambient_light=float(data.get("ambient_light", 1.0)),
            surface_friction_mult=float(data.get("surface_friction_mult", 1.0)),
            target_speed_override=float(data["target_speed_override"]) if data.get("target_speed_override") is not None else None,
            time_limit_override=float(data["time_limit_override"]) if data.get("time_limit_override") is not None else None,
            sensor_noise_mult=float(data.get("sensor_noise_mult", 1.0)),
            spawn_override=dict(data["spawn_override"]) if data.get("spawn_override") else None,
            obstacle_overrides=list(data.get("obstacle_overrides", [])),
            randomization=rand_def
        )

    @classmethod
    def get_standard_scenarios(cls) -> Dict[str, ScenarioDefinition]:
        """Built-in library of reusable scenarios."""
        return {
            "basic_lane_following": cls(
                scenario_id="basic_lane_following",
                name="Basic Lane Following",
                description="Clean dry conditions, standard lighting, focusing on centerline tracking.",
                weather="clear",
                time_of_day="day",
                ambient_light=1.0,
                surface_friction_mult=1.0,
                target_speed_override=18.0
            ),
            "high_speed_racing": cls(
                scenario_id="high_speed_racing",
                name="High Speed Racing",
                description="High target speed corridor demanding precise steering control.",
                weather="clear",
                time_of_day="day",
                ambient_light=1.0,
                surface_friction_mult=1.0,
                target_speed_override=35.0
            ),
            "wet_adverse_weather": cls(
                scenario_id="wet_adverse_weather",
                name="Wet Track Adverse Weather",
                description="Low-friction wet asphalt with reduced ambient lighting.",
                weather="rain",
                time_of_day="dusk",
                ambient_light=0.6,
                surface_friction_mult=0.65,
                target_speed_override=16.0
            ),
            "obstacle_evasion": cls(
                scenario_id="obstacle_evasion",
                name="Obstacle Evasion",
                description="Course with static barriers and cones requiring evasive maneuvering.",
                weather="clear",
                time_of_day="day",
                ambient_light=1.0,
                surface_friction_mult=1.0,
                obstacle_overrides=[
                    {"name": "Evasion Cone 1", "entity_type": "cone", "pos": [30.0, 10.0, 0.0], "yaw": 0.0},
                    {"name": "Evasion Barrier", "entity_type": "barrier", "pos": [-25.0, -15.0, 0.0], "yaw": 0.4},
                ]
            ),
            "sensor_noise_challenge": cls(
                scenario_id="sensor_noise_challenge",
                name="Sensor Noise Challenge",
                description="Elevated LiDAR and IMU noise forcing policy robustness.",
                weather="fog",
                time_of_day="night",
                ambient_light=0.3,
                sensor_noise_mult=2.5,
                surface_friction_mult=0.9
            ),
            "full_domain_randomization": cls(
                scenario_id="full_domain_randomization",
                name="Full Domain Randomization",
                description="Active randomized friction, mass, and spawn perturbations on every reset.",
                weather="clear",
                time_of_day="day",
                ambient_light=1.0,
                randomization=DomainRandomizationDefinition(
                    enabled=True,
                    global_seed=1337,
                    parameters=DomainRandomizationDefinition.create_default().parameters
                )
            ),
        }
