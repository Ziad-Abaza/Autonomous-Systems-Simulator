"""
Unit tests for Termination Designer, Gymnasium semantics (terminated vs truncated),
condition rules, cause attribution, and serialization.
"""

import pytest
import math

from sim_env.termination_designer import (
    TerminationDefinition,
    TerminationRuleConfig,
    TerminationConditionType,
    CompiledTerminationEvaluator
)


def test_termination_defaults_and_evaluation():
    td = TerminationDefinition.create_default_racing_termination()
    evaluator = td.compile_evaluator()
    evaluator.reset()

    # Normal running state
    term, trunc, reason = evaluator.evaluate(
        dt=0.0166, is_colliding=False, is_on_road=True,
        heading_error=0.0, checkpoint_passed=False, laps_completed=0
    )
    assert term is False
    assert trunc is False
    assert reason["reason"] == "running"


def test_collision_termination_is_not_truncation():
    td = TerminationDefinition.create_default_racing_termination()
    evaluator = td.compile_evaluator()
    evaluator.reset()

    term, trunc, reason = evaluator.evaluate(
        dt=0.0166, is_colliding=True, is_on_road=True,
        heading_error=0.0, checkpoint_passed=False, laps_completed=0
    )
    assert term is True
    assert trunc is False
    assert reason["reason"] == "collision"
    assert reason["is_truncation"] is False


def test_offroad_and_wrong_direction_terminations():
    td = TerminationDefinition.create_default_racing_termination()
    evaluator = td.compile_evaluator()

    # Off-road
    evaluator.reset()
    term, trunc, reason = evaluator.evaluate(
        dt=0.0166, is_colliding=False, is_on_road=False,
        heading_error=0.0, checkpoint_passed=False, laps_completed=0
    )
    assert term is True
    assert trunc is False
    assert reason["reason"] == "off_road"

    # Wrong direction (> 120 deg)
    evaluator.reset()
    term, trunc, reason = evaluator.evaluate(
        dt=0.0166, is_colliding=False, is_on_road=True,
        heading_error=math.radians(130.0), checkpoint_passed=False, laps_completed=0
    )
    assert term is True
    assert trunc is False
    assert reason["reason"] == "wrong_direction"


def test_max_steps_is_truncation_not_termination():
    td = TerminationDefinition(rules=[
        TerminationRuleConfig(
            rule_id="trunc_steps",
            name="Max Steps",
            condition_type=TerminationConditionType.MAX_STEPS,
            is_truncation=True,
            params={"max_steps": 5}
        )
    ])
    evaluator = td.compile_evaluator()
    evaluator.reset()

    for step in range(4):
        term, trunc, _ = evaluator.evaluate(
            dt=0.0166, is_colliding=False, is_on_road=True,
            heading_error=0.0, checkpoint_passed=False, laps_completed=0
        )
        assert term is False and trunc is False

    # Step 5 reaches max_steps
    term, trunc, reason = evaluator.evaluate(
        dt=0.0166, is_colliding=False, is_on_road=True,
        heading_error=0.0, checkpoint_passed=False, laps_completed=0
    )
    assert term is False
    assert trunc is True
    assert reason["is_truncation"] is True
    assert reason["step"] == 5


def test_stuck_checkpoint_timeout_truncation():
    td = TerminationDefinition(rules=[
        TerminationRuleConfig(
            rule_id="trunc_stuck",
            name="Checkpoint Timeout",
            condition_type=TerminationConditionType.CHECKPOINT_TIMEOUT,
            is_truncation=True,
            params={"max_seconds": 2.0}
        )
    ])
    evaluator = td.compile_evaluator()
    evaluator.reset()

    # Step without checkpoint for 1.5 seconds -> still running
    term, trunc, _ = evaluator.evaluate(
        dt=1.5, is_colliding=False, is_on_road=True,
        heading_error=0.0, checkpoint_passed=False, laps_completed=0
    )
    assert term is False and trunc is False

    # Pass checkpoint -> timer resets
    term, trunc, _ = evaluator.evaluate(
        dt=0.1, is_colliding=False, is_on_road=True,
        heading_error=0.0, checkpoint_passed=True, laps_completed=0
    )
    assert term is False and trunc is False
    assert evaluator.time_since_checkpoint == 0.0

    # Step for 2.1 seconds without checkpoint -> truncation triggers
    term, trunc, reason = evaluator.evaluate(
        dt=2.1, is_colliding=False, is_on_road=True,
        heading_error=0.0, checkpoint_passed=False, laps_completed=0
    )
    assert term is False
    assert trunc is True
    assert reason["is_truncation"] is True
