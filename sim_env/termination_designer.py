"""
Termination and Truncation Designer.
Provides composable termination rules, strict Gymnasium semantics
(terminated vs truncated), and machine-readable termination cause attribution.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional


class TerminationConditionType:
    COLLISION = "collision"
    OFF_ROAD = "off_road"
    WRONG_DIRECTION = "wrong_direction"
    COURSE_COMPLETION = "course_completion"
    MAX_STEPS = "max_steps"
    SIMULATION_TIMEOUT = "simulation_timeout"
    CHECKPOINT_TIMEOUT = "checkpoint_timeout"
    CUSTOM_THRESHOLD = "custom_threshold"


@dataclass
class TerminationRuleConfig:
    """
    Specification for a single termination or truncation rule.
    """
    rule_id: str
    name: str
    condition_type: str
    enabled: bool = True
    is_truncation: bool = False       # If True -> truncated=True; If False -> terminated=True
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TerminationRuleConfig:
        return cls(
            rule_id=str(data.get("rule_id", "rule")),
            name=str(data.get("name", "Termination Rule")),
            condition_type=str(data.get("condition_type", TerminationConditionType.COLLISION)),
            enabled=bool(data.get("enabled", True)),
            is_truncation=bool(data.get("is_truncation", False)),
            params=dict(data.get("params", {})),
            description=str(data.get("description", ""))
        )


@dataclass
class TerminationDefinition:
    """
    User-configurable collection of termination and truncation rules.
    """
    rules: List[TerminationRuleConfig] = field(default_factory=list)

    @classmethod
    def create_default_racing_termination(cls) -> TerminationDefinition:
        """Standard automotive RL termination rules."""
        rules = [
            TerminationRuleConfig(
                rule_id="term_collision",
                name="Barrier Collision",
                condition_type=TerminationConditionType.COLLISION,
                enabled=True,
                is_truncation=False,  # Task failure -> terminated
                description="Terminates immediately when vehicle contacts barrier or obstacle"
            ),
            TerminationRuleConfig(
                rule_id="term_off_road",
                name="Off-Road Excursion",
                condition_type=TerminationConditionType.OFF_ROAD,
                enabled=True,
                is_truncation=False,  # Task failure -> terminated
                description="Terminates when vehicle fully departs road surface"
            ),
            TerminationRuleConfig(
                rule_id="term_wrong_direction",
                name="Wrong Direction",
                condition_type=TerminationConditionType.WRONG_DIRECTION,
                enabled=True,
                is_truncation=False,  # Task failure -> terminated
                params={"max_angle_deg": 120.0},
                description="Terminates if heading angle exceeds 120° relative to track"
            ),
            TerminationRuleConfig(
                rule_id="term_completion",
                name="Lap Completion",
                condition_type=TerminationConditionType.COURSE_COMPLETION,
                enabled=False,
                is_truncation=False,  # Task success -> terminated
                params={"target_laps": 1},
                description="Terminates with success when target laps are completed"
            ),
            TerminationRuleConfig(
                rule_id="trunc_max_steps",
                name="Step Limit",
                condition_type=TerminationConditionType.MAX_STEPS,
                enabled=True,
                is_truncation=True,   # Horizon limit -> truncated
                params={"max_steps": 5000},
                description="Truncates episode when maximum step limit is reached"
            ),
            TerminationRuleConfig(
                rule_id="trunc_stuck",
                name="Stuck Timeout",
                condition_type=TerminationConditionType.CHECKPOINT_TIMEOUT,
                enabled=True,
                is_truncation=True,   # Horizon limit -> truncated
                params={"max_seconds": 30.0},
                description="Truncates episode if no checkpoint is passed within time window"
            ),
        ]
        return cls(rules=rules)

    def add_rule(self, rule: TerminationRuleConfig) -> None:
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        init_len = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < init_len

    def get_rule(self, rule_id: str) -> Optional[TerminationRuleConfig]:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"rules": [r.to_dict() for r in self.rules]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TerminationDefinition:
        raw = data.get("rules", [])
        return cls(rules=[TerminationRuleConfig.from_dict(r) for r in raw])

    def compile_evaluator(self) -> CompiledTerminationEvaluator:
        return CompiledTerminationEvaluator(self)


class CompiledTerminationEvaluator:
    """
    Optimized runtime evaluator for termination rules.
    Outputs strict (terminated, truncated, machine_readable_reason).
    """
    def __init__(self, definition: TerminationDefinition):
        self.definition = definition
        self.active_rules = [r for r in definition.rules if r.enabled]
        self.episode_steps: int = 0
        self.episode_time: float = 0.0
        self.time_since_checkpoint: float = 0.0

    def reset(self) -> None:
        self.episode_steps = 0
        self.episode_time = 0.0
        self.time_since_checkpoint = 0.0

    def evaluate(
        self,
        dt: float,
        is_colliding: bool,
        is_on_road: bool,
        heading_error: float,
        checkpoint_passed: bool,
        laps_completed: int,
        is_closed: bool = True
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Evaluates active rules in priority order.
        Returns: (terminated: bool, truncated: bool, reason_dict: Dict[str, Any])
        """
        self.episode_steps += 1
        self.episode_time += dt

        if checkpoint_passed:
            self.time_since_checkpoint = 0.0
        else:
            self.time_since_checkpoint += dt

        for rule in self.active_rules:
            rtype = rule.condition_type
            triggered = False

            if rtype == TerminationConditionType.COLLISION:
                triggered = is_colliding

            elif rtype == TerminationConditionType.OFF_ROAD:
                triggered = not is_on_road

            elif rtype == TerminationConditionType.WRONG_DIRECTION:
                max_deg = rule.params.get("max_angle_deg", 120.0)
                triggered = abs(heading_error) > math.radians(max_deg)

            elif rtype == TerminationConditionType.COURSE_COMPLETION:
                target_laps = rule.params.get("target_laps", 1)
                if not is_closed and laps_completed >= 1:
                    triggered = True
                elif is_closed and laps_completed >= target_laps:
                    triggered = True

            elif rtype == TerminationConditionType.MAX_STEPS:
                max_steps = rule.params.get("max_steps", 5000)
                triggered = (max_steps > 0) and (self.episode_steps >= max_steps)

            elif rtype == TerminationConditionType.SIMULATION_TIMEOUT:
                max_sec = rule.params.get("max_seconds", 60.0)
                triggered = (max_sec > 0.0) and (self.episode_time >= max_sec)

            elif rtype == TerminationConditionType.CHECKPOINT_TIMEOUT:
                max_sec = rule.params.get("max_seconds", 30.0)
                triggered = (max_sec > 0.0) and (self.time_since_checkpoint >= max_sec)

            if triggered:
                is_trunc = rule.is_truncation
                reason_dict = {
                    "rule_id": rule.rule_id,
                    "condition": rule.condition_type,
                    "reason": rule.rule_id.replace("term_", "").replace("trunc_", ""),
                    "is_truncation": is_trunc,
                    "step": self.episode_steps,
                    "sim_time": round(self.episode_time, 4)
                }
                if is_trunc:
                    return False, True, reason_dict
                else:
                    return True, False, reason_dict

        return False, False, {"reason": "running", "step": self.episode_steps, "sim_time": round(self.episode_time, 4)}
