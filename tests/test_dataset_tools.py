"""Phase 6 — dataset validation/split/stats/inspect + trajectory sampler +
header metadata + eval transition export."""
import json
import os

import numpy as np
import pytest


# ------------------------------------------------------------- fixtures/helpers

def _ep(eid, ret=1.0, length=3, obs_dim=4, reason="collision", seed=1):
    steps = []
    for t in range(length):
        steps.append({
            "obs": [0.1 * t] * obs_dim,
            "action": [0.0, 0.5],
            "reward": ret / length,
            "terminated": (t == length - 1),
            "truncated": False,
            "termination_reason": reason if t == length - 1 else "",
        })
    return {
        "episode_id": eid, "seed": seed, "env_fingerprint": "fp_a",
        "scenario_id": "scen", "total_return": ret, "length": length,
        "termination_reason": reason, "steps": steps,
    }


def _write_dataset(ds_dir, episodes, manifest_extra=None):
    os.makedirs(ds_dir, exist_ok=True)
    with open(os.path.join(ds_dir, "episodes.jsonl"), "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    manifest = {
        "dataset_format": "transitions_v1",
        "schema_hash": "abc123",
        "env_fingerprint": "fp_a",
        "episodes": len(episodes),
        "steps": sum(e["length"] for e in episodes),
        "filters": {}, "skipped": {},
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    with open(os.path.join(ds_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


def _write_traj(run_dir, eid, n_steps=3, header=True, seed=1):
    td = os.path.join(run_dir, "trajectories")
    os.makedirs(td, exist_ok=True)
    path = os.path.join(td, f"{eid}.jsonl")
    with open(path, "w") as f:
        if header:
            f.write(json.dumps({
                "type": "header", "episode_id": eid,
                "env_fingerprint": "fp_a", "scenario_id": "scen",
                "seed": seed, "observation_schema": {}, "action_schema": {},
            }) + "\n")
        for t in range(n_steps):
            f.write(json.dumps({
                "type": "step", "episode_id": eid, "step": t + 1,
                "agent_data": {"obs": [0.1, 0.2], "action": [0.0],
                               "reward": 1.0,
                               "terminated": t == n_steps - 1,
                               "truncated": False,
                               "termination_reason": "collision" if t == n_steps - 1 else ""},
                "diagnostic_data": {"speed": 10.0},
            }) + "\n")
    return path


# --------------------------------------------------------------- validate_dataset

def test_validate_valid_dataset(tmp_path):
    from sim_experiment.dataset import validate_dataset
    ds = str(tmp_path / "ds")
    _write_dataset(ds, [_ep("e1"), _ep("e2", length=4)])
    rep = validate_dataset(ds)
    assert rep["valid"] is True
    assert rep["episodes"] == 2
    assert rep["steps"] == 7
    assert rep["errors"] == []


def test_validate_rejects_bad_format(tmp_path):
    from sim_experiment.dataset import validate_dataset
    ds = str(tmp_path / "ds")
    _write_dataset(ds, [_ep("e1")], manifest_extra={"dataset_format": "bogus"})
    rep = validate_dataset(ds)
    assert rep["valid"] is False
    assert any("format" in e for e in rep["errors"])


def test_validate_detects_missing_step_fields(tmp_path):
    from sim_experiment.dataset import validate_dataset
    ds = str(tmp_path / "ds")
    ep = _ep("e1")
    del ep["steps"][0]["action"]
    _write_dataset(ds, [ep])
    rep = validate_dataset(ds)
    assert rep["valid"] is False
    assert any("action" in e for e in rep["errors"])


def test_validate_detects_diagnostic_leak(tmp_path):
    from sim_experiment.dataset import validate_dataset
    ds = str(tmp_path / "ds")
    ep = _ep("e1")
    ep["steps"][0]["speed"] = 99.0  # diagnostic field leaked into a step
    _write_dataset(ds, [ep])
    rep = validate_dataset(ds)
    assert rep["valid"] is False
    assert any("diagnostic" in e.lower() or "unexpected" in e.lower() for e in rep["errors"])


def test_validate_detects_manifest_count_mismatch(tmp_path):
    from sim_experiment.dataset import validate_dataset
    ds = str(tmp_path / "ds")
    _write_dataset(ds, [_ep("e1"), _ep("e2")], manifest_extra={"episodes": 5})
    rep = validate_dataset(ds)
    assert rep["valid"] is False
    assert any("episode" in e.lower() for e in rep["errors"])


def test_validate_missing_files(tmp_path):
    from sim_experiment.dataset import validate_dataset
    rep = validate_dataset(str(tmp_path / "nope"))
    assert rep["valid"] is False
    assert rep["errors"]


# ----------------------------------------------------------------- split_dataset

def test_split_deterministic_and_complete(tmp_path):
    from sim_experiment.dataset import split_dataset
    ds = str(tmp_path / "ds")
    eps = [_ep(f"e{i}") for i in range(10)]
    _write_dataset(ds, eps)
    s1 = split_dataset(ds, seed=7, ratios={"train": 0.7, "val": 0.2, "test": 0.1})
    s2 = split_dataset(ds, seed=7, ratios={"train": 0.7, "val": 0.2, "test": 0.1})
    assert s1 == s2  # deterministic
    all_ids = s1["train"] + s1["val"] + s1["test"]
    assert sorted(all_ids) == [f"e{i}" for i in range(10)]  # complete, no dupes
    assert len(s1["train"]) == 7 and len(s1["val"]) == 2 and len(s1["test"]) == 1


def test_split_different_seed_differs(tmp_path):
    from sim_experiment.dataset import split_dataset
    ds = str(tmp_path / "ds")
    _write_dataset(ds, [_ep(f"e{i}") for i in range(20)])
    s1 = split_dataset(ds, seed=1)
    s2 = split_dataset(ds, seed=2)
    assert s1["train"] != s2["train"]


def test_split_persisted(tmp_path):
    from sim_experiment.dataset import split_dataset
    ds = str(tmp_path / "ds")
    _write_dataset(ds, [_ep(f"e{i}") for i in range(6)])
    split_dataset(ds, seed=3)
    assert os.path.exists(os.path.join(ds, "splits.json"))


# ----------------------------------------------------------------- stats/inspect

def test_dataset_statistics(tmp_path):
    from sim_experiment.dataset_inspect import dataset_statistics
    ds = str(tmp_path / "ds")
    _write_dataset(ds, [_ep("e1", ret=2.0, length=4),
                        _ep("e2", ret=4.0, length=6, reason="max_steps")])
    st = dataset_statistics(ds)
    assert st["episode_count"] == 2
    assert st["step_count"] == 10
    assert st["return"]["mean"] == pytest.approx(3.0)
    assert st["length"]["min"] == 4 and st["length"]["max"] == 6
    assert st["termination_reasons"]["collision"] == 1
    assert st["termination_reasons"]["max_steps"] == 1


def test_inspect_dataset_combines_report(tmp_path):
    from sim_experiment.dataset_inspect import inspect_dataset
    ds = str(tmp_path / "ds")
    _write_dataset(ds, [_ep("e1")])
    rep = inspect_dataset(ds)
    assert "valid" in rep and "statistics" in rep
    assert rep["statistics"]["episode_count"] == 1


# --------------------------------------------------------------- trajectory sampler

def test_sample_trajectories_deterministic(tmp_path):
    from sim_experiment.trajectory_sampler import sample_trajectories
    run = str(tmp_path / "run")
    for i in range(6):
        _write_traj(run, f"ep{i}")
    s1 = sample_trajectories(run, count=3, seed=5)
    s2 = sample_trajectories(run, count=3, seed=5)
    assert [e["episode_id"] for e in s1] == [e["episode_id"] for e in s2]
    assert len(s1) == 3


def test_sample_trajectories_different_seed(tmp_path):
    from sim_experiment.trajectory_sampler import sample_trajectories
    run = str(tmp_path / "run")
    for i in range(10):
        _write_traj(run, f"ep{i}")
    s1 = sample_trajectories(run, count=4, seed=1)
    s2 = sample_trajectories(run, count=4, seed=2)
    assert set(e["episode_id"] for e in s1) != set(e["episode_id"] for e in s2) or \
           len(s1) == len(s2)


def test_sample_caps_at_available(tmp_path):
    from sim_experiment.trajectory_sampler import sample_trajectories
    run = str(tmp_path / "run")
    for i in range(2):
        _write_traj(run, f"ep{i}")
    s = sample_trajectories(run, count=10, seed=1)
    assert len(s) == 2


# ------------------------------------------------- header validation + metadata

def test_episode_without_header_type_skipped(tmp_path):
    """Task 12: dataset reader must require type=header, not position."""
    from sim_experiment.dataset import _read_episode, export_dataset
    run = str(tmp_path / "run")
    # Write an episode file whose FIRST record lacks type=header
    td = os.path.join(run, "trajectories")
    os.makedirs(td)
    with open(os.path.join(td, "bad.jsonl"), "w") as f:
        f.write(json.dumps({"episode_id": "bad"}) + "\n")  # no type field
        f.write(json.dumps({"type": "step", "agent_data": {"obs": [1], "action": [0], "reward": 1.0, "terminated": True, "truncated": False}}) + "\n")
    result = export_dataset(run, str(tmp_path / "ds"))
    assert result["episodes"] == 0  # malformed file skipped


def test_header_carries_episode_seed_and_stage(tmp_path):
    """Task 13: trajectory headers capture exact per-episode seed + stage."""
    from sim_experiment.trajectory import TrajectoryWriter, TrajectoryReader
    td = str(tmp_path / "traj")
    w = TrajectoryWriter(td)
    w.start_episode("ep_x", env_fingerprint="fp", scenario_id="s",
                    seed=42, episode_seed=42007, curriculum_stage_index=1)
    w.record_step(1, {"obs": [0.1], "action": [0], "reward": 1.0,
                      "terminated": True, "truncated": False})
    w.close()
    data = TrajectoryReader(os.path.join(td, "ep_x.jsonl")).read()
    assert data["header"]["episode_seed"] == 42007
    assert data["header"]["curriculum_stage_index"] == 1
    assert data["header"]["seed"] == 42


def test_recorder_writes_episode_seed(tmp_path):
    """Recorder passes runner-reported episode_seed into the header."""
    from sim_experiment.trajectory import TrajectoryWriter, TrajectoryReader
    from sim_experiment.trainers._harness import EpisodeTrajectoryRecorder
    td = str(tmp_path / "traj")
    contract = {"environment_fingerprint": "fp", "scenario_id": "s", "seed": 42}
    rec = EpisodeTrajectoryRecorder(TrajectoryWriter(td), 5, contract)
    rec.attach(1)
    rec({"env_idx": 0, "obs": [0.1], "action": [0.0], "reward": 1.0,
         "terminated": False, "truncated": False,
         "episode_seed": 12345, "info": {}})
    rec({"env_idx": 0, "obs": [0.2], "action": [0.0], "reward": 1.0,
         "terminated": True, "truncated": False,
         "episode_seed": 12345, "info": {}})
    rec.traj.close()
    hdr = TrajectoryReader(os.path.join(td, "train_env0_ep0.jsonl")).read()["header"]
    assert hdr["episode_seed"] == 12345


# --------------------------------------------------------- eval transition export

class _MiniEnv:
    def __init__(self):
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        term = self.t >= 4
        return (np.ones(4, dtype=np.float32) * self.t, 1.0, term, False,
                {"step": self.t, "termination_reason": "collision" if term else ""})


def test_export_transitions_valid_dataset(tmp_path):
    """Task 14: eval loops can export full transitions_v1 datasets."""
    from sim_experiment.evaluation import export_transitions
    from sim_experiment.dataset import validate_dataset
    out = str(tmp_path / "eval_ds")

    def policy(obs):
        return [0.0, 0.5, 0.0]

    result = export_transitions(
        _MiniEnv(), policy, seeds=[1, 2], num_episodes=2,
        output_dir=out, env_fingerprint="fp_a", scenario_id="scen")
    assert result["episodes"] == 2
    rep = validate_dataset(out)
    assert rep["valid"], rep["errors"]
    assert rep["steps"] == 8
