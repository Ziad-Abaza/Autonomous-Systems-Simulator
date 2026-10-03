"""
Curriculum Learning Configuration.
Defines deterministic sequential training stages, scenario assignments,
environmental overrides, and progression thresholds.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class CurriculumStage:
    """
    Specification for a single stage in an RL curriculum.
    """
    stage_id: int
    name: str
    description: str
    scenario_id: str
    target_metric: str = "mean_return"          # "mean_return", "lap_completion_rate", "collision_rate"
    advancement_threshold: float = 100.0        # Value required to advance to next stage
    min_episodes: int = 50                      # Minimum episodes before advancement is permitted
    environment_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CurriculumStage:
        return cls(
            stage_id=int(data.get("stage_id", 1)),
            name=str(data.get("name", "Stage")),
            description=str(data.get("description", "")),
            scenario_id=str(data.get("scenario_id", "basic_lane_following")),
            target_metric=str(data.get("target_metric", "mean_return")),
            advancement_threshold=float(data.get("advancement_threshold", 100.0)),
            min_episodes=int(data.get("min_episodes", 50)),
            environment_overrides=dict(data.get("environment_overrides", {}))
        )


@dataclass
class CurriculumDefinition:
    """
    User-configurable multi-stage curriculum pipeline.
    """
    name: str = "Standard Autonomous Driving Curriculum"
    stages: List[CurriculumStage] = field(default_factory=list)
    current_stage_idx: int = 0

    @classmethod
    def create_default(cls) -> CurriculumDefinition:
        stages = [
            CurriculumStage(
                stage_id=1,
                name="Stage 1: Lane Keeping Basics",
                description="Low target speed, wide road, zero obstacles. Master basic centering.",
                scenario_id="basic_lane_following",
                target_metric="mean_return",
                advancement_threshold=50.0,
                min_episodes=20,
                environment_overrides={"target_speed": 12.0}
            ),
            CurriculumStage(
                stage_id=2,
                name="Stage 2: High Speed Cruising",
                description="Elevated speed requiring smooth steering and braking into curves.",
                scenario_id="high_speed_racing",
                target_metric="mean_return",
                advancement_threshold=120.0,
                min_episodes=30,
                environment_overrides={"target_speed": 22.0}
            ),
            CurriculumStage(
                stage_id=3,
                name="Stage 3: Obstacle Navigation",
                description="Cones and barricades introduced requiring LiDAR awareness.",
                scenario_id="obstacle_evasion",
                target_metric="lap_completion_rate",
                advancement_threshold=0.85,
                min_episodes=40,
            ),
            CurriculumStage(
                stage_id=4,
                name="Stage 4: Adverse Weather & Low Friction",
                description="Wet asphalt and rain with reduced surface grip.",
                scenario_id="wet_adverse_weather",
                target_metric="lap_completion_rate",
                advancement_threshold=0.80,
                min_episodes=40,
            ),
            CurriculumStage(
                stage_id=5,
                name="Stage 5: Full Domain Randomization",
                description="Varying mass, tire friction, and spawn jitter for ultimate generalization.",
                scenario_id="full_domain_randomization",
                target_metric="lap_completion_rate",
                advancement_threshold=0.90,
                min_episodes=50,
            ),
        ]
        return cls(stages=stages, current_stage_idx=0)

    def get_current_stage(self) -> Optional[CurriculumStage]:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    def advance_stage(self) -> bool:
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "current_stage_idx": self.current_stage_idx,
            "stages": [s.to_dict() for s in self.stages]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CurriculumDefinition:
        raw_stages = data.get("stages", [])
        stages = [CurriculumStage.from_dict(s) for s in raw_stages]
        return cls(
            name=str(data.get("name", "Curriculum")),
            current_stage_idx=int(data.get("current_stage_idx", 0)),
            stages=stages
        )
