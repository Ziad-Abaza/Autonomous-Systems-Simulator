"""Phase 6 — VectorEnv abstraction: SyncVectorEnv semantics."""
import numpy as np
import pytest

from sim_experiment.vec_env import SyncVectorEnv, is_vec_env


class ScriptedEnv:
    """Minimal gym-style env; records seeds + scenario for assertions."""

    def __init__(self, obs_dim=8, term_at=None, reward=1.0):
        self.term_at = term_at
        self.reward = reward
        self.obs_dim = obs_dim
        self.t = 0
        self.reset_seeds = []
        self.scenario = None
        self.closed = False

    def reset(self, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.t = 0
        return np.zeros(self.obs_dim, dtype=np.float32) + (seed or 0), {}

    def step(self, action):
        self.t += 1
        term = self.term_at is not None and self.t >= self.term_at
        info = {"step": self.t, "termination_reason": "x" if term else ""}
        return (np.ones(self.obs_dim, dtype=np.float32) * self.t,
                self.reward, term, False, info)

    def set_scenario(self, scenario_def):
        self.scenario = scenario_def

    def close(self):
        self.closed = True


def test_is_vec_env():
    envs = [ScriptedEnv(), ScriptedEnv()]
    assert is_vec_env(SyncVectorEnv(envs))
    assert not is_vec_env(envs)
    assert not is_vec_env(ScriptedEnv())


def test_num_envs_and_reset_all_seeds():
    envs = [ScriptedEnv() for _ in range(3)]
    vec = SyncVectorEnv(envs)
    assert vec.num_envs == 3
    obs = vec.reset_all([10, 11, 12])
    assert len(obs) == 3
    assert [e.reset_seeds for e in envs] == [[10], [11], [12]]
    # seeds flow into obs via the scripted env -> independent streams
    assert obs[0][0] == 10.0 and obs[1][0] == 11.0 and obs[2][0] == 12.0


def test_step_all_returns_per_env_tuples():
    envs = [ScriptedEnv(term_at=3), ScriptedEnv(term_at=5)]
    vec = SyncVectorEnv(envs)
    vec.reset_all([1, 2])
    r1 = vec.step_all([[0, 0, 0], [0, 0, 0]])
    r2 = vec.step_all([[0, 0, 0], [0, 0, 0]])
    assert len(r1) == 2 and len(r2) == 2
    for res in (r1, r2):
        for tup in res:
            assert len(tup) == 5  # obs, reward, terminated, truncated, info
    r3 = vec.step_all([[0, 0, 0], [0, 0, 0]])
    assert r3[0][2] is True or r3[0][2] == True   # env0 terminated at t=3
    assert r3[1][2] is False or r3[1][2] == False  # env1 still running


def test_reset_at_single_env():
    envs = [ScriptedEnv(), ScriptedEnv()]
    vec = SyncVectorEnv(envs)
    vec.reset_all([1, 2])
    vec.step_all([[0, 0, 0], [0, 0, 0]])
    vec.reset_at(0, seed=99)
    assert envs[0].reset_seeds == [1, 99]
    assert envs[1].reset_seeds == [2]
    assert envs[0].t == 0 and envs[1].t == 1


def test_set_scenario_broadcast():
    envs = [ScriptedEnv(), ScriptedEnv()]
    vec = SyncVectorEnv(envs)
    vec.set_scenario({"name": "wet_adverse_weather"})
    for e in envs:
        assert e.scenario.name == "wet_adverse_weather"


def test_set_scenario_env_without_api_raises():
    class Bare:
        pass

    vec = SyncVectorEnv([ScriptedEnv()])
    vec.envs = [Bare()]
    with pytest.raises(NotImplementedError):
        vec.set_scenario({"name": "x"})


def test_close_closes_all_idempotent():
    envs = [ScriptedEnv(), ScriptedEnv()]
    vec = SyncVectorEnv(envs)
    vec.close()
    vec.close()
    assert all(e.closed for e in envs)


def test_independent_trajectories():
    """N envs stepped via step_all produce N independent episode streams."""
    envs = [ScriptedEnv(term_at=2, reward=1.0), ScriptedEnv(term_at=4, reward=2.0)]
    vec = SyncVectorEnv(envs)
    vec.reset_all([7, 8])
    terms = []
    for _ in range(4):
        res = vec.step_all([[0, 0, 0], [0, 0, 0]])
        terms.append([bool(r[2]) for r in res])
    # env0 first terminates at step 2; env1 at step 4 — separate lifecycles
    assert [t[0] for t in terms] == [False, True, True, True]
    assert [t[1] for t in terms] == [False, False, False, True]
