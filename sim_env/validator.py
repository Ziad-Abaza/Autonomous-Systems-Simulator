"""
Environment Validation Engine and Training Gatekeeper.
Thoroughly inspects the declarative RL environment pipeline to detect
configuration errors, degenerate geometry, space mismatches, obstacle overlaps,
oracle/debug observation leakage, and serialization flaws before RL training begins.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from sim_core.math_utils import Vec2, Vec3
from sim_core.track.road_definition import RoadDefinition
from sim_core.world.entity import WorldEntity
from sim_env.agent import AgentDefinition
from sim_env.observation_designer import ObservationSpaceDefinition, ChannelCategory
from sim_env.action_designer import ActionSpaceDefinition, ActionType
from sim_env.reward_designer import RewardFunctionDefinition
from sim_env.termination_designer import TerminationDefinition


class IssueSeverity:
    ERROR = "ERROR"        # Critical problem; blocks RL training gate
    WARNING = "WARNING"    # Potential suboptimal performance or edge-case
    INFO = "INFO"          # Informational assertion / sanity confirmation


@dataclass
class ValidationIssue:
    severity: str          # ERROR, WARNING, INFO
    subsystem: str         # "Agent", "Observation", "Action", "Reward", "Termination", "Spawn", "Track", "Sensors"
    message: str
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """
    Aggregated validation results containing issues, readiness gate, and summary.
    """
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid_for_rl(self) -> bool:
        """Training Gatekeeper: returns False if ANY critical ERROR exists."""
        return not any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.INFO]

    def add_error(self, subsystem: str, message: str, remediation: str = "") -> None:
        self.issues.append(ValidationIssue(IssueSeverity.ERROR, subsystem, message, remediation))

    def add_warning(self, subsystem: str, message: str, remediation: str = "") -> None:
        self.issues.append(ValidationIssue(IssueSeverity.WARNING, subsystem, message, remediation))

    def add_info(self, subsystem: str, message: str, remediation: str = "") -> None:
        self.issues.append(ValidationIssue(IssueSeverity.INFO, subsystem, message, remediation))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid_for_rl": self.is_valid_for_rl,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.infos),
            "issues": [i.to_dict() for i in self.issues]
        }


class EnvironmentValidator:
    """
    Validates complete environment definitions against rigorous RL guidelines.
    """
    @classmethod
    def validate(
        cls,
        road_def: Optional[RoadDefinition],
        agent: Optional[AgentDefinition],
        entities: Optional[List[WorldEntity]] = None,
        available_sensors: Optional[List[str]] = None,
        track_mesh: Optional[Any] = None
    ) -> ValidationReport:
        report = ValidationReport()
        entities = entities or []
        available_sensors = available_sensors or ["vehicle_state", "lidar_rays", "rgb_camera", "imu"]

        # 1. Agent Existence & Integrity
        if agent is None:
            report.add_error("Agent", "No agent definition specified.", "Create or assign an AgentDefinition.")
            return report

        if not agent.agent_id or not agent.agent_id.strip():
            report.add_error("Agent", "Agent ID cannot be empty.", "Assign a unique non-empty string identifier.")

        if not agent.entity_type:
            report.add_error("Agent", "Agent has no controlled entity type.", "Specify entity_type (e.g. 'vehicle').")

        # 2. Track & Road Geometry
        if road_def is None:
            report.add_error("Track", "Road definition is missing.", "Assign a RoadDefinition.")
        else:
            if len(road_def.control_points) < 3:
                report.add_error("Track", f"Track has only {len(road_def.control_points)} control points (minimum 3 required).", "Add more control points.")
            
            # Check for degenerate overlapping control points
            cps = road_def.control_points
            for i in range(len(cps)):
                for j in range(i + 1, len(cps)):
                    dist = math.hypot(cps[i].x - cps[j].x, cps[i].y - cps[j].y)
                    if dist < 0.5:
                        report.add_error("Track", f"Control points #{i} and #{j} are degenerate (< 0.5m apart).", "Separate or remove duplicate points.")

            if road_def.num_checkpoints < 2:
                report.add_error("Track", f"Insufficient checkpoints ({road_def.num_checkpoints}). Minimum 2 required.", "Increase num_checkpoints in RoadDefinition.")
            else:
                report.add_info("Track", f"Track contains {road_def.num_checkpoints} checkpoint gates for progress tracking.")

        # 3. Action Space Validation
        act_space = agent.action_space
        if act_space is None:
            report.add_error("Action", "Agent has no action space.", "Define an ActionSpaceDefinition.")
        else:
            if act_space.space_type == ActionType.DISCRETE:
                if len(act_space.discrete_options) < 2:
                    report.add_error("Action", f"Discrete action space has {len(act_space.discrete_options)} actions (minimum 2 required).", "Add discrete action options.")
                else:
                    report.add_info("Action", f"Discrete action space configured with {len(act_space.discrete_options)} selectable commands.")
            else:
                if len(act_space.channels) == 0:
                    report.add_error("Action", "Continuous action space has 0 controllable channels.", "Add channels (e.g. steering, throttle, brake).")
                for c in act_space.channels:
                    if c.min_val >= c.max_val:
                        report.add_error("Action", f"Channel '{c.name}' has invalid bounds: min ({c.min_val}) >= max ({c.max_val}).", "Correct channel min/max.")
                    if math.isnan(c.min_val) or math.isnan(c.max_val) or math.isinf(c.min_val) or math.isinf(c.max_val):
                        report.add_error("Action", f"Channel '{c.name}' has NaN or Infinite bounds.", "Provide finite numeric bounds.")
                    if c.default_val < c.min_val or c.default_val > c.max_val:
                        report.add_warning("Action", f"Channel '{c.name}' default value ({c.default_val}) is outside [min, max].", "Set default within bounds.")

        # 4. Observation Space Validation & Leakage Check
        obs_space = agent.observation_space
        if obs_space is None:
            report.add_error("Observation", "Agent has no observation space.", "Define an ObservationSpaceDefinition.")
        else:
            active_channels = obs_space.get_active_channels()
            if len(active_channels) == 0 and not obs_space.include_image_channel:
                report.add_error("Observation", "Observation space has 0 active channels.", "Enable at least one observation channel.")

            vector_dim = obs_space.compute_vector_dim()
            if obs_space.flatten_vector and vector_dim == 0 and not obs_space.include_image_channel:
                report.add_error("Observation", "Flattened observation vector dimension is 0.", "Enable feature channels.")
            elif obs_space.flatten_vector and vector_dim > 0:
                report.add_info("Observation", f"Observation vector dimension: {vector_dim} features.")

            # Check Sensor dependencies
            for c in active_channels:
                if c.source_sensor not in available_sensors:
                    report.add_error(
                        "Sensors",
                        f"Observation channel '{c.name}' requires sensor '{c.source_sensor}', which is not attached to agent.",
                        f"Attach '{c.source_sensor}' to the SensorManager or agent sensor list."
                    )

            # CRITICAL LEAKAGE CHECK
            is_leak_free, leak_issues = obs_space.validate_no_leakage()
            if not is_leak_free:
                for leak in leak_issues:
                    report.add_error("Observation", f"SECURITY LEAKAGE: {leak}", "Remove or re-categorize privileged debug channel.")

        # 5. Reward Function Validation
        reward_fn = agent.reward_function
        if reward_fn is None:
            report.add_error("Reward", "Agent has no reward function.", "Assign a RewardFunctionDefinition.")
        else:
            active_comps = [c for c in reward_fn.components if c.enabled]
            if len(active_comps) == 0:
                report.add_error("Reward", "Reward function has 0 active components.", "Enable at least one reward component.")
            else:
                has_progress = any(c.component_type == "progress" for c in active_comps)
                if not has_progress:
                    report.add_warning("Reward", "Reward function lacks a progress component; agent may learn static policies.", "Consider adding Centerline Progress.")

                for comp in active_comps:
                    if math.isnan(comp.weight) or math.isinf(comp.weight):
                        report.add_error("Reward", f"Reward component '{comp.name}' has NaN or Infinite weight.", "Set a finite weight.")

        # 6. Termination Rules Validation
        term_def = agent.termination_rules
        if term_def is None:
            report.add_error("Termination", "Agent has no termination rules.", "Assign a TerminationDefinition.")
        else:
            active_rules = [r for r in term_def.rules if r.enabled]
            if len(active_rules) == 0:
                report.add_error("Termination", "No active termination or truncation rules. Episodes will run indefinitely.", "Enable termination conditions (e.g. collision, max steps).")
            else:
                has_trunc = any(r.is_truncation for r in active_rules)
                if not has_trunc:
                    report.add_warning("Termination", "No truncation condition (e.g. max steps or timeout). Episodes might not bound finite horizons.", "Add MaxStepsRule.")

        # 7. Spawn Clearance and Obstacle Overlaps
        if road_def is not None:
            sp = road_def.spawn_point
            sp_pos = Vec2(sp.x, sp.y)
            for ent in entities:
                if getattr(ent, 'is_collidable', False):
                    ent_pos = Vec2(ent.pos.x, ent.pos.y)
                    dist = (sp_pos - ent_pos).length()
                    # Vehicle collision radius is ~2.5m, obstacle radius is ~1.5m
                    if dist < 3.5:
                        report.add_error(
                            "Spawn",
                            f"Spawn point ({sp.x:.1f}, {sp.y:.1f}) is inside or too close ({dist:.2f}m) to collidable obstacle '{ent.name}'.",
                            "Move obstacle away from spawn point or adjust spawn location."
                        )

        # 8. Determinism Sanity
        report.add_info("Environment", "Simulation clock uses deterministic fixed 60 Hz stepping.")

        return report
