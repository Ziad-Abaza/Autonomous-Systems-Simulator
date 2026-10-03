"""
Curriculum Runtime Controller.

Consumes a CurriculumDefinition (the declarative model in sim_env.curriculum)
and drives deterministic stage advancement during a training run. This lives
in the experiment layer — no curriculum logic is added to sim_core or the
vehicle model.

Advancement rule (deterministic):
    A stage advances only when ALL of the following hold at evaluation time:
      - the stage's target metric is present in the evaluation aggregate
      - episodes_in_stage >= stage.min_episodes  (training episodes counted
        inside the stage — never a single lucky episode)
      - the metric value reaches the threshold (>= for maximize metrics,
        <= for lower-is-better metrics like collision_rate)

State is persisted via to_state()/from_state() so checkpoints and run state
can restore the exact stage, progress, and history on resume.
"""

from __future__ import annotations
import hashlib
import json
import time
from typing import Dict, Any, List, Optional

from sim_env.curriculum import CurriculumDefinition, CurriculumStage
from sim_env.scenario_designer import ScenarioDefinition

CURRICULUM_STATE_VERSION = "1.0"

# Curriculum stage metric names -> evaluation aggregate keys.
# "mean_return" and "lap_completion_rate" are the names used by the Phase 3
# CurriculumDefinition defaults; the aggregate names are the canonical form.
_METRIC_ALIASES: Dict[str, str] = {
    "mean_return": "mean_reward",
    "mean_reward": "mean_reward",
    "lap_completion_rate": "completion_rate",
    "completion_rate": "completion_rate",
    "collision_rate": "collision_rate",
    "off_road_rate": "off_road_rate",
    "timeout_rate": "timeout_rate",
    "mean_episode_length": "mean_episode_length",
    "mean_lateral_error": "mean_lateral_error",
    "mean_heading_error": "mean_heading_error",
    "mean_speed": "mean_speed",
}

# Metrics where a smaller value is better — threshold compare inverts.
_LOWER_IS_BETTER = {
    "collision_rate", "off_road_rate", "timeout_rate",
    "mean_lateral_error", "mean_heading_error",
}

# CurriculumStage.environment_overrides keys -> ScenarioDefinition fields.
_OVERRIDE_KEY_MAP = {
    "target_speed": "target_speed_override",
    "time_limit": "time_limit_override",
    "surface_friction_mult": "surface_friction_mult",
    "sensor_noise_mult": "sensor_noise_mult",
    "ambient_light": "ambient_light",
    "weather": "weather",
    "time_of_day": "time_of_day",
}


def curriculum_fingerprint(curriculum_dict: Optional[Dict[str, Any]]) -> str:
    """Deterministic identity for a curriculum definition (empty if none)."""
    if not curriculum_dict:
        return ""
    return hashlib.sha256(
        json.dumps(curriculum_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _resolve_metric_name(target_metric: str) -> Optional[str]:
    return _METRIC_ALIASES.get(target_metric, target_metric if target_metric else None)


def validate_curriculum(
    curriculum_dict: Optional[Dict[str, Any]],
    known_scenario_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Structural validation of a serialized CurriculumDefinition.
    Returns a list of problems; empty means valid.
    """
    if not curriculum_dict:
        return []
    errors: List[str] = []
    stages = curriculum_dict.get("stages")
    if not isinstance(stages, list) or not stages:
        errors.append("curriculum has no stages")
        return errors

    # Standard scenario library plus any caller-supplied extras (e.g. the
    # experiment's own scenario_id).
    known = set(ScenarioDefinition.get_standard_scenarios().keys())
    known.update(known_scenario_ids or [])
    seen_ids = set()
    for i, s in enumerate(stages):
        sid = s.get("scenario_id", "")
        if sid and sid not in known:
            errors.append(f"stage {i}: unknown scenario_id '{sid}'")
        metric = s.get("target_metric", "")
        if _resolve_metric_name(metric) not in set(_METRIC_ALIASES.values()):
            errors.append(f"stage {i}: unknown target_metric '{metric}'")
        if int(s.get("min_episodes", 0)) < 1:
            errors.append(f"stage {i}: min_episodes must be >= 1")
        thr = s.get("advancement_threshold")
        if not isinstance(thr, (int, float)):
            errors.append(f"stage {i}: advancement_threshold must be numeric")
        stid = s.get("stage_id", i)
        if stid in seen_ids:
            errors.append(f"stage {i}: duplicate stage_id {stid}")
        seen_ids.add(stid)
    return errors


class CurriculumController:
    """
    Runtime state machine over a CurriculumDefinition.
    Tracks the active stage, episode progress within it, evaluation-driven
    advancement decisions, and a full audit history.
    """

    def __init__(
        self,
        definition: CurriculumDefinition,
        base_seed: int = 42,
        stage_index: int = 0,
        episodes_in_stage: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        base_scenario_dict: Optional[Dict[str, Any]] = None,
    ):
        self.definition = definition
        self.base_seed = int(base_seed)
        self.stage_index = int(stage_index)
        self.episodes_in_stage = int(episodes_in_stage)
        self.history: List[Dict[str, Any]] = list(history or [])
        self._base_scenario_dict = dict(base_scenario_dict or {})
        self._standard = ScenarioDefinition.get_standard_scenarios()

    # ------------------------------------------------------------ stage info

    @property
    def is_complete(self) -> bool:
        """True when on the final stage (no further advancement possible)."""
        return self.stage_index >= len(self.definition.stages) - 1

    @property
    def current_stage(self) -> Optional[CurriculumStage]:
        if 0 <= self.stage_index < len(self.definition.stages):
            return self.definition.stages[self.stage_index]
        return None

    @property
    def next_stage(self) -> Optional[CurriculumStage]:
        nxt = self.stage_index + 1
        if nxt < len(self.definition.stages):
            return self.definition.stages[nxt]
        return None

    def stage_seed(self, env_index: int = 0) -> int:
        """Deterministic per-stage, per-env seed derivation."""
        return self.base_seed + int(env_index) + self.stage_index * 10000

    def stage_scenario_dict(self) -> Dict[str, Any]:
        """
        Resolves the active stage's scenario: the standard scenario snapshot
        (or the manifest scenario if ids match) with stage
        environment_overrides applied onto ScenarioDefinition fields.
        """
        stage = self.current_stage
        if stage is None:
            return dict(self._base_scenario_dict)
        base = self._standard.get(stage.scenario_id)
        if base is not None:
            d = base.to_dict()
        elif self._base_scenario_dict.get("scenario_id") == stage.scenario_id:
            d = dict(self._base_scenario_dict)
        else:
            raise ValueError(
                f"Curriculum stage '{stage.name}' references unknown scenario "
                f"'{stage.scenario_id}'"
            )
        for key, value in (stage.environment_overrides or {}).items():
            field = _OVERRIDE_KEY_MAP.get(key, key)
            d[field] = value
        return d

    # ------------------------------------------------------------ runtime

    def record_training_episodes(self, count: int) -> None:
        self.episodes_in_stage += int(count)

    def _metric_passes(self, metric_name: str, value: float, threshold: float) -> bool:
        if metric_name in _LOWER_IS_BETTER:
            return value <= threshold
        return value >= threshold

    def evaluate_advancement(
        self, aggregate: Dict[str, Any], timestep: int = 0
    ) -> Dict[str, Any]:
        """
        Consumes an evaluation aggregate and decides whether to advance.
        Always returns a decision record; never raises for missing metrics.
        """
        stage = self.current_stage
        if stage is None:
            rec = {"stage": None, "advanced": False, "reason": "curriculum_complete",
                   "timestep": int(timestep), "ts": round(time.time(), 4)}
            self.history.append(rec)
            return rec

        metric_key = _resolve_metric_name(stage.target_metric)
        value = aggregate.get(metric_key) if metric_key else None
        available = isinstance(value, (int, float))
        enough_eps = self.episodes_in_stage >= stage.min_episodes
        can_advance = self.next_stage is not None
        passed = (
            available and enough_eps and can_advance
            and self._metric_passes(metric_key, float(value), float(stage.advancement_threshold))
        )

        rec = {
            "stage": stage.name,
            "stage_index": self.stage_index,
            "stage_id": stage.stage_id,
            "scenario_id": stage.scenario_id,
            "metric": stage.target_metric,
            "resolved_metric": metric_key,
            "metric_value_available": bool(available),
            "value": float(value) if available else None,
            "threshold": float(stage.advancement_threshold),
            "direction": "min" if metric_key in _LOWER_IS_BETTER else "max",
            "episodes": self.episodes_in_stage,
            "min_episodes": stage.min_episodes,
            "timestep": int(timestep),
            "advanced": bool(passed),
            "ts": round(time.time(), 4),
        }
        self.history.append(rec)

        if passed:
            self.stage_index += 1
            self.episodes_in_stage = 0
            nxt = self.current_stage
            rec["next_stage"] = nxt.name if nxt else None
            rec["next_stage_index"] = self.stage_index
        return rec

    # ------------------------------------------------------------ persistence

    def to_state(self) -> Dict[str, Any]:
        stage = self.current_stage
        return {
            "version": CURRICULUM_STATE_VERSION,
            "curriculum_name": self.definition.name,
            "curriculum_fingerprint": curriculum_fingerprint(self.definition.to_dict()),
            "stage_index": self.stage_index,
            "stage_id": stage.stage_id if stage else None,
            "stage_name": stage.name if stage else None,
            "episodes_in_stage": self.episodes_in_stage,
            "is_complete": self.is_complete,
            "base_seed": self.base_seed,
            "history": list(self.history),
        }

    @classmethod
    def from_state(
        cls,
        definition: CurriculumDefinition,
        state: Dict[str, Any],
        base_scenario_dict: Optional[Dict[str, Any]] = None,
    ) -> "CurriculumController":
        return cls(
            definition=definition,
            base_seed=int(state.get("base_seed", 42)),
            stage_index=int(state.get("stage_index", 0)),
            episodes_in_stage=int(state.get("episodes_in_stage", 0)),
            history=list(state.get("history", [])),
            base_scenario_dict=base_scenario_dict,
        )
