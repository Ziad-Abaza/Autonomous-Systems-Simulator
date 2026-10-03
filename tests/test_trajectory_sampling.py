"""Phase 6 — predicate-driven trajectory sampling in EpisodeTrajectoryRecorder."""
import os

import pytest

from sim_experiment.trajectory import TrajectoryWriter, TrajectoryReader


def _contract(sampling=None, traj_eps=None):
    alg = {}
    if sampling:
        alg["trajectory_sampling"] = sampling
    if traj_eps is not None:
        alg["trajectory_episodes"] = traj_eps
    return {
        "environment_fingerprint": "fp", "scenario_id": "s", "seed": 42,
        "training": {"algorithm_config": alg},
    }


def _recorder(tmp_path, contract, limit=3):
    from sim_experiment.trainers._harness import EpisodeTrajectoryRecorder
    rec = EpisodeTrajectoryRecorder(TrajectoryWriter(str(tmp_path)),
                                    limit, contract)
    rec.attach(1)
    return rec


def _step(env_idx=0, reward=1.0, done=False, reason="collision", seed=100):
    return {
        "env_idx": env_idx, "obs": [0.1], "action": [0.0],
        "reward": reward, "terminated": done, "truncated": False,
        "episode_seed": seed, "info": {"termination_reason": reason if done else ""},
    }


def _files(tmp_path):
    return sorted(f for f in os.listdir(str(tmp_path)) if f.endswith(".jsonl"))


def _run_episodes(rec, n, reward=1.0, reason="collision"):
    for ep in range(n):
        rec(_step(done=False))
        rec(_step(done=True, reward=reward, reason=reason))
    rec.traj.close()


def test_first_n_default(tmp_path):
    """Back-compat: trajectory_episodes=N == {mode:first_n, n:N}."""
    rec = _recorder(tmp_path, _contract(traj_eps=2), limit=2)
    _run_episodes(rec, 4)
    assert _files(tmp_path) == ["train_env0_ep0.jsonl", "train_env0_ep1.jsonl"]


def test_every_n(tmp_path):
    rec = _recorder(tmp_path,
                    _contract(sampling={"mode": "every_n", "n": 2}), limit=10)
    _run_episodes(rec, 6)
    # episodes 0, 2, 4
    assert _files(tmp_path) == ["train_env0_ep0.jsonl", "train_env0_ep2.jsonl",
                                "train_env0_ep4.jsonl"]


def test_probability_deterministic(tmp_path):
    cfg = _contract(sampling={"mode": "probability", "p": 0.5})
    r1 = _recorder(tmp_path / "a", cfg, limit=99)
    _run_episodes(r1, 8)
    files1 = _files(tmp_path / "a")
    r2 = _recorder(tmp_path / "b", cfg, limit=99)
    _run_episodes(r2, 8)
    assert files1 == _files(tmp_path / "b")  # deterministic
    assert 0 < len(files1) < 8  # subset selected by hash


def test_episode_ids_filter(tmp_path):
    rec = _recorder(tmp_path, _contract(sampling={
        "mode": "episodes", "episode_ids": ["train_env0_ep1", "train_env0_ep3"]}),
        limit=99)
    _run_episodes(rec, 5)
    assert _files(tmp_path) == ["train_env0_ep1.jsonl", "train_env0_ep3.jsonl"]


def test_min_return_filter(tmp_path):
    rec = _recorder(tmp_path,
                    _contract(sampling={"mode": "min_return", "min_return": 5.0}),
                    limit=99)
    # ep0 low return, ep1 high return
    rec(_step(reward=1.0))
    rec(_step(done=True, reward=1.0))
    rec(_step(reward=10.0))
    rec(_step(done=True, reward=10.0))
    rec.traj.close()
    assert _files(tmp_path) == ["train_env0_ep1.jsonl"]


def test_termination_reasons_filter(tmp_path):
    rec = _recorder(tmp_path, _contract(sampling={
        "mode": "termination_reasons",
        "termination_reasons": ["lap_completed"]}), limit=99)
    rec(_step())
    rec(_step(done=True, reason="collision"))
    rec(_step())
    rec(_step(done=True, reason="lap_completed"))
    rec.traj.close()
    assert _files(tmp_path) == ["train_env0_ep1.jsonl"]


def test_all_mode(tmp_path):
    rec = _recorder(tmp_path, _contract(sampling={"mode": "all"}), limit=99)
    _run_episodes(rec, 3)
    assert len(_files(tmp_path)) == 3


def test_no_partial_files_on_drop(tmp_path):
    """Dropped episodes leave no file at all (buffered write on episode end)."""
    rec = _recorder(tmp_path,
                    _contract(sampling={"mode": "first_n", "n": 1}), limit=1)
    _run_episodes(rec, 3)
    files = _files(tmp_path)
    assert files == ["train_env0_ep0.jsonl"]
    # Recorded file is complete and parseable.
    data = TrajectoryReader(os.path.join(str(tmp_path), files[0])).read()
    assert data["header"]["type"] == "header"
    assert len(data["steps"]) == 2
