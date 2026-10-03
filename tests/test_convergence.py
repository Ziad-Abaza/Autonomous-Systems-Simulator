"""Phase 6 — learning benchmark: convergence report + benchmark runner."""
import json
import os

import numpy as np
import pytest


def _results(phase_rewards, seed_std=None):
    """Synthetic benchmark results: list of per-phase mean rewards."""
    phases = []
    for i, r in enumerate(phase_rewards):
        name = "baseline" if i == 0 else f"phase{i}"
        rewards = {"1": r, "2": r}
        phases.append({
            "name": name,
            "eval": {
                "aggregate": {
                    "mean_reward": r,
                    "completion_rate": min(1.0, max(0.0, (r + 5) / 20)),
                    "episode_count": 2,
                },
                "per_seed": rewards,
            },
        })
    return {"phases": phases, "config": {"thresholds": {}}}


# ----------------------------------------------------------- convergence_report

def test_verdict_improved():
    from sim_experiment.convergence import convergence_report
    rep = convergence_report(
        _results([0.0, 5.0, 10.0]),
        min_reward_delta=1.0, stability_tolerance=2.0)
    assert rep["verdict"] == "improved"
    assert rep["improvement"]["delta_mean_reward"] == pytest.approx(10.0)
    assert rep["baseline_mean_reward"] == pytest.approx(0.0)
    assert rep["final_mean_reward"] == pytest.approx(10.0)


def test_verdict_no_improvement():
    from sim_experiment.convergence import convergence_report
    rep = convergence_report(
        _results([5.0, 5.5, 5.8]),
        min_reward_delta=1.0, stability_tolerance=2.0)
    assert rep["verdict"] == "no_improvement"


def test_verdict_regressed():
    from sim_experiment.convergence import convergence_report
    rep = convergence_report(
        _results([10.0, 8.0, 5.0]),
        min_reward_delta=1.0, stability_tolerance=2.0)
    assert rep["verdict"] == "regressed"


def test_stability_flags_regression_phase():
    from sim_experiment.convergence import convergence_report
    # Phase1 improves, phase2 drops beyond tolerance but still above baseline.
    rep = convergence_report(
        _results([0.0, 10.0, 7.0]),
        min_reward_delta=1.0, stability_tolerance=2.0)
    assert rep["stable"] is False
    assert rep["verdict"] == "unstable"
    assert rep["regressions"]


def test_cross_seed_stats():
    from sim_experiment.convergence import convergence_report
    rep = convergence_report(_results([0.0, 4.0, 8.0]),
                             min_reward_delta=1.0, stability_tolerance=2.0)
    assert "cross_seed" in rep
    assert rep["cross_seed"]["final"]["mean"] == pytest.approx(8.0)


def test_single_phase_no_verdict():
    from sim_experiment.convergence import convergence_report
    rep = convergence_report(_results([3.0]),
                             min_reward_delta=1.0, stability_tolerance=1.0)
    assert rep["verdict"] == "insufficient_data"


# ------------------------------------------------------------- benchmark runner

def test_config_hash_deterministic():
    from benchmarks.phase6.benchmark_runner import _config_hash
    cfg = {"a": 1, "b": [1, 2], "c": {"x": "y"}}
    assert _config_hash(cfg) == _config_hash(dict(cfg))
    assert _config_hash(cfg) != _config_hash({"a": 2})


def test_run_benchmark_end_to_end(tmp_path):
    """Real two-phase PPO benchmark: baseline eval -> train -> eval -> resume -> eval."""
    from benchmarks.phase6.benchmark_runner import run_benchmark
    from sim_experiment.convergence import convergence_report

    config = {
        "name": "bench_test",
        "template": "lane_following",
        "trainer": "ppo",
        "random_seed": 42,
        "eval_seeds": [11, 22],
        "num_eval_episodes": 1,
        "phases": [{"timesteps": 64, "rollout_length": 64},
                   {"timesteps": 64, "rollout_length": 64}],
        "training_overrides": {"num_envs": 1, "eval_frequency": 0,
                               "checkpoint_frequency": 0},
        "thresholds": {"min_reward_delta": 0.0, "stability_tolerance": 1e9},
    }
    results = run_benchmark(config, work_dir=str(tmp_path))

    assert results["config_hash"]
    assert results["environment"]["sim_version"]
    assert len(results["phases"]) == 3  # baseline + 2 training phases
    assert results["phases"][0]["name"] == "baseline"
    for ph in results["phases"][1:]:
        assert ph["run"]["status"] == "COMPLETED"
        assert ph["checkpoint"]
        assert ph["sps"] > 0
        assert "mean_reward" in ph["eval"]["aggregate"]

    rep = convergence_report(results)
    assert rep["verdict"] in ("improved", "no_improvement", "unstable")

    # Results persist as JSON for the report surface.
    out = os.path.join(results["work_dir"], "benchmark_results.json")
    assert os.path.exists(out)
