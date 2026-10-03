"""Phase 6 — ProcessVectorEnv: real process-isolated parallel envs."""
import numpy as np
import pytest

from sim_experiment.process_env import ProcessVectorEnv, EnvWorkerCrash
from sim_experiment.vec_env import is_vec_env


# ------------------------------------------------------------------ worker
# Module-level so multiprocessing spawn can pickle the callable.

class _ScriptedEnv:
    def __init__(self, spec, seed):
        self.term_at = spec.get("term_at")
        self.reward = spec.get("reward", 1.0)
        self.obs_dim = spec.get("obs_dim", 4)
        self.seed = seed
        self.t = 0
        self.reset_seeds = []

    def reset(self, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.t = 0
        return np.zeros(self.obs_dim, dtype=np.float32) + float(seed or 0), {}

    def step(self, action):
        self.t += 1
        term = self.term_at is not None and self.t >= self.term_at
        return (np.ones(self.obs_dim, dtype=np.float32) * self.t,
                self.reward, term, False,
                {"step": self.t, "termination_reason": "collision" if term else ""})

    def set_scenario(self, scenario_def):
        self.scenario = scenario_def


def _scripted_worker(conn, env_dict, scenario_dict, seed):
    env = _ScriptedEnv(env_dict, seed)
    conn.send("ready")
    while True:
        try:
            cmd, payload = conn.recv()
        except EOFError:
            break
        if cmd == "close":
            break
        if cmd == "reset":
            conn.send(("ok", env.reset(seed=payload)[0]))
        elif cmd == "step":
            conn.send(("ok", env.step(payload)))
        elif cmd == "set_scenario":
            env.scenario = payload
            conn.send(("ok", None))
        elif cmd == "reset_seeds":
            conn.send(("ok", list(env.reset_seeds)))
    conn.close()


def _pool(n=2, spec=None, seeds=(1, 2)):
    return ProcessVectorEnv(
        env_dict=spec or {"term_at": 3, "obs_dim": 4},
        scenario_dict=None,
        seeds=list(seeds[:n]),
        worker_fn=_scripted_worker,
    )


# --------------------------------------------------------------------- tests

def test_is_vec_env_and_num_envs():
    pool = _pool(2)
    try:
        assert is_vec_env(pool)
        assert pool.num_envs == 2
        assert pool.kind == "process"
    finally:
        pool.close()


def test_reset_all_deterministic_seeds():
    pool = _pool(2, seeds=(10, 20))
    try:
        obs = pool.reset_all([10, 20])
        assert obs[0][0] == 10.0 and obs[1][0] == 20.0
        obs2 = pool.reset_all([10, 20])
        assert np.array_equal(obs[0], obs2[0]) and np.array_equal(obs[1], obs2[1])
    finally:
        pool.close()


def test_step_all_parallel_results():
    pool = _pool(2, spec={"term_at": 2})
    try:
        pool.reset_all([1, 2])
        r1 = pool.step_all([[0, 0, 0], [0, 0, 0]])
        r2 = pool.step_all([[0, 0, 0], [0, 0, 0]])
        assert len(r1) == 2 and len(r2) == 2
        assert all(len(t) == 5 for t in r1)
        assert r2[0][2] is True and r2[1][2] is True  # both term at t=2
    finally:
        pool.close()


def test_reset_at_single_env():
    pool = _pool(2)
    try:
        pool.reset_all([1, 2])
        pool.step_all([[0, 0, 0], [0, 0, 0]])
        pool.reset_at(0, seed=77)
        seeds = pool._rpc_one(0, "reset_seeds")
        assert seeds == [1, 77]
    finally:
        pool.close()


def test_set_scenario_broadcast():
    pool = _pool(2)
    try:
        pool.set_scenario({"name": "x"})
    finally:
        pool.close()


def test_worker_crash_isolated_and_detected():
    pool = _pool(2)
    try:
        pool.reset_all([1, 2])
        pool._procs[0].terminate()
        pool._procs[0].join(timeout=10)
        with pytest.raises(EnvWorkerCrash):
            pool.step_all([[0, 0, 0], [0, 0, 0]])
        # sibling env still functional
        out = pool.reset_at(1, seed=9)
        assert out[0] == 9.0
    finally:
        pool.close()


def test_close_is_clean():
    pool = _pool(2)
    pool.close()
    pool.close()
    assert all(not p.is_alive() for p in pool._procs)


# ------------------------------------------------------- real environment E2E

def test_real_env_in_process_pool():
    """End-to-end: real SimulationEnvironment built inside worker processes."""
    from sim_env.templates import EnvironmentTemplateManager

    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    env_dict = project.to_dict()
    pool = ProcessVectorEnv(env_dict=env_dict, scenario_dict=None, seeds=[3, 4])
    try:
        obs = pool.reset_all([3, 4])
        assert len(obs) == 2
        results = pool.step_all([[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]])
        for o, r, term, trunc, info in results:
            assert isinstance(float(r), float)
            assert "step" in info
        # deterministic: identical seeds -> identical first obs
        obs_b = pool.reset_all([3, 4])
        assert np.allclose(np.asarray(obs[0]["vector"] if isinstance(obs[0], dict) else obs[0]),
                           np.asarray(obs_b[0]["vector"] if isinstance(obs_b[0], dict) else obs_b[0]))
    finally:
        pool.close()
