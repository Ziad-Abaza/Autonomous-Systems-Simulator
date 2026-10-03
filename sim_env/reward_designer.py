"""
Reward Designer and Composable Reward Engine.
Provides modular authorable reward components, parameter validation,
falloff curve models, live decomposed debugging, and safety guards
against future information leakage, teleportation, and checkpoint exploitation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, Callable


class FalloffType:
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"


def _motion_gate(speed: float, params: Dict[str, Any]) -> float:
    """Scales a shaping reward by vehicle motion.

    Alignment terms (centering, heading) reward *maintaining* lane
    discipline, which is only meaningful while the vehicle is actually
    moving. Without a gate a parked vehicle collects full alignment
    credit (~0.8/step) — the stationary-policy exploit. When
    ``motion_gate_ms`` is set, the raw component value is scaled by
    ``min(1, speed / gate)`` so alignment credit requires real motion.
    A value of 0.0 or absence disables the gate (legacy semantics).
    """
    gate = float(params.get("motion_gate_ms", 0.0) or 0.0)
    if gate <= 0.0:
        return 1.0
    return min(1.0, max(0.0, speed / gate))


@dataclass
class RewardComponentConfig:
    """
    Specification for a single composable reward component.
    """
    component_id: str
    name: str
    component_type: str
    weight: float = 1.0
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RewardComponentConfig:
        return cls(
            component_id=str(data.get("component_id", "reward_comp")),
            name=str(data.get("name", "Reward Component")),
            component_type=str(data.get("component_type", "progress")),
            weight=float(data.get("weight", 1.0)),
            enabled=bool(data.get("enabled", True)),
            params=dict(data.get("params", {})),
            description=str(data.get("description", ""))
        )


@dataclass
class RewardFunctionDefinition:
    """
    User-configurable collection of reward components forming a reward graph.
    """
    components: List[RewardComponentConfig] = field(default_factory=list)

    @classmethod
    def create_default_racing_reward(cls) -> RewardFunctionDefinition:
        """
        Creates standard balanced vehicle racing reward function.
        """
        comps = [
            RewardComponentConfig(
                component_id="progress",
                name="Centerline Progress",
                component_type="progress",
                weight=1.0,
                params={"max_step_delta_m": 5.0},
                description="Reward per meter of forward progress along track centerline"
            ),
            RewardComponentConfig(
                component_id="centering",
                name="Centerline Deviation",
                component_type="centerline",
                weight=0.5,
                params={"max_distance_m": 6.0, "falloff": FalloffType.LINEAR,
                        "motion_gate_ms": 5.0},
                description="Reward for holding track centerline while moving (1.0 at center; gated below 5 m/s)"
            ),
            RewardComponentConfig(
                component_id="speed",
                name="Target Speed",
                component_type="speed",
                weight=0.2,
                params={"target_speed_ms": 20.0, "tolerance": 5.0},
                description="Reward for matching and maintaining target speed"
            ),
            RewardComponentConfig(
                component_id="heading",
                name="Heading Alignment",
                component_type="heading",
                weight=0.3,
                params={"motion_gate_ms": 5.0},
                description="Reward for holding track heading while moving (cos(heading_error); gated below 5 m/s)"
            ),
            RewardComponentConfig(
                component_id="smooth_steer",
                name="Steering Smoothness",
                component_type="smooth_steer",
                weight=-0.05,
                params={},
                description="Penalty on squared change in steering angle between consecutive ticks"
            ),
            RewardComponentConfig(
                component_id="checkpoint",
                name="Checkpoint Crossing",
                component_type="checkpoint",
                weight=10.0,
                params={},
                description="Bonus awarded when legally passing a forward checkpoint gate"
            ),
            RewardComponentConfig(
                component_id="completion",
                name="Lap Completion",
                component_type="completion",
                weight=100.0,
                params={},
                description="Bonus awarded upon completing a full circuit lap or open track"
            ),
            RewardComponentConfig(
                component_id="collision",
                name="Collision Penalty",
                component_type="collision",
                weight=-50.0,
                params={},
                description="Penalty on colliding with track barrier or obstacle"
            ),
            RewardComponentConfig(
                component_id="off_road",
                name="Off-Road Penalty",
                component_type="off_road",
                weight=-25.0,
                params={},
                description="Penalty on leaving the drivable road surface"
            ),
            RewardComponentConfig(
                component_id="reverse",
                name="Reverse Driving Penalty",
                component_type="reverse",
                weight=-1.0,
                params={"heading_threshold_deg": 100.0},
                description="Penalty per step for driving backward or pointing wrong direction"
            ),
            RewardComponentConfig(
                component_id="time_penalty",
                name="Time Step Penalty",
                component_type="time_penalty",
                weight=-0.01,
                params={},
                description="Small per-step cost encouraging fast course completion"
            ),
        ]
        return cls(components=comps)

    def add_component(self, comp: RewardComponentConfig) -> None:
        self.components.append(comp)

    def remove_component(self, component_id: str) -> bool:
        init_len = len(self.components)
        self.components = [c for c in self.components if c.component_id != component_id]
        return len(self.components) < init_len

    def get_component(self, component_id: str) -> Optional[RewardComponentConfig]:
        for c in self.components:
            if c.component_id == component_id:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"components": [c.to_dict() for c in self.components]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RewardFunctionDefinition:
        raw = data.get("components", [])
        return cls(components=[RewardComponentConfig.from_dict(c) for c in raw])

    def export_graph(self) -> Dict[str, Any]:
        """Exports graph nodes and weights representation for UI visualization."""
        nodes = []
        for c in self.components:
            nodes.append({
                "id": c.component_id,
                "label": c.name,
                "type": c.component_type,
                "weight": c.weight,
                "enabled": c.enabled,
                "params": c.params
            })
        return {
            "nodes": nodes,
            "aggregation": "weighted_sum"
        }

    def compile_engine(self) -> CompiledRewardEngine:
        """Compiles definition into zero-overhead runtime evaluation engine."""
        return CompiledRewardEngine(self)


class CompiledRewardEngine:
    """
    High-performance runtime reward engine with full decomposition,
    teleportation guards, and strict current-step information constraints.
    """
    def __init__(self, definition: RewardFunctionDefinition):
        self.definition = definition
        self.active_components = [c for c in definition.components if c.enabled]

        # Prior step state
        self.prev_s: float = 0.0
        self.prev_steer: float = 0.0
        self.prev_accel: float = 0.0
        self.total_accumulated_reward: float = 0.0

        # Detailed breakdown of last step: {comp_id: {"raw": float, "weight": float, "contrib": float}}
        self.last_detailed_breakdown: Dict[str, Dict[str, float]] = {}
        self.last_breakdown: Dict[str, float] = {}

    def reset(self, initial_s: float = 0.0, initial_steer: float = 0.0) -> None:
        self.prev_s = initial_s
        self.prev_steer = initial_steer
        self.prev_accel = 0.0
        self.total_accumulated_reward = 0.0
        self.last_detailed_breakdown = {}
        self.last_breakdown = {c.component_id: 0.0 for c in self.active_components}
        self.last_breakdown["total"] = 0.0

    def compute_step_reward(
        self,
        current_s: float,
        track_length: float,
        is_closed: bool,
        lateral_offset: float,
        road_width: float,
        speed: float,
        heading_error: float,
        current_steer: float,
        is_colliding: bool,
        is_on_road: bool,
        checkpoint_passed: bool,
        lap_completed: bool,
        dt: float = 1.0 / 60.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes the total step reward from active components.
        Operates strictly on current and prior step telemetry.
        """
        step_total = 0.0
        breakdown_simple: Dict[str, float] = {}
        breakdown_detailed: Dict[str, Dict[str, float]] = {}

        # 1. Delta S calculation with wrap-around and teleportation guard
        delta_s = current_s - self.prev_s
        if is_closed and track_length > 0:
            if delta_s < -track_length * 0.5:
                delta_s += track_length
            elif delta_s > track_length * 0.5:
                delta_s -= track_length

        # Teleportation guard: ignore jumps > 5.0m per tick (impossible at realistic speeds)
        if abs(delta_s) > 5.0:
            delta_s = 0.0

        # Delta steering calculation
        delta_steer = current_steer - self.prev_steer

        # Evaluate each active component
        for comp in self.active_components:
            ctype = comp.component_type
            raw_val = 0.0

            if ctype == "progress":
                raw_val = delta_s

            elif ctype == "centerline":
                max_dist = comp.params.get("max_distance_m", max(1.0, road_width * 0.5))
                falloff = comp.params.get("falloff", FalloffType.LINEAR)
                norm_dist = min(1.0, abs(lateral_offset) / max(0.1, max_dist))

                if falloff == FalloffType.QUADRATIC:
                    raw_val = max(0.0, 1.0 - norm_dist * norm_dist)
                elif falloff == FalloffType.EXPONENTIAL:
                    raw_val = math.exp(-3.0 * norm_dist)
                else:  # LINEAR
                    raw_val = max(0.0, 1.0 - norm_dist)
                raw_val *= _motion_gate(speed, comp.params)

            elif ctype == "speed":
                target = comp.params.get("target_speed_ms", 20.0)
                norm_speed = min(1.0, max(0.0, speed / max(1.0, target)))
                raw_val = norm_speed

            elif ctype == "heading":
                raw_val = math.cos(heading_error) * _motion_gate(speed, comp.params)

            elif ctype == "smooth_steer":
                raw_val = -(delta_steer * delta_steer)

            elif ctype == "checkpoint":
                raw_val = 1.0 if checkpoint_passed else 0.0

            elif ctype == "completion":
                raw_val = 1.0 if lap_completed else 0.0

            elif ctype == "collision":
                raw_val = 1.0 if is_colliding else 0.0

            elif ctype == "off_road":
                raw_val = 1.0 if not is_on_road else 0.0

            elif ctype == "reverse":
                thresh_rad = math.radians(comp.params.get("heading_threshold_deg", 100.0))
                is_backward = (delta_s < -0.05) or (abs(heading_error) > thresh_rad)
                raw_val = 1.0 if is_backward else 0.0

            elif ctype == "time_penalty":
                raw_val = 1.0

            # Calculate contribution
            contribution = raw_val * comp.weight
            step_total += contribution

            breakdown_simple[comp.component_id] = float(contribution)
            breakdown_detailed[comp.component_id] = {
                "raw": float(raw_val),
                "weight": float(comp.weight),
                "contrib": float(contribution)
            }

        # Update historical state
        self.prev_s = current_s
        self.prev_steer = current_steer

        breakdown_simple["total"] = float(step_total)
        self.last_breakdown = breakdown_simple
        self.last_detailed_breakdown = breakdown_detailed
        self.total_accumulated_reward += step_total

        return float(step_total), breakdown_simple
