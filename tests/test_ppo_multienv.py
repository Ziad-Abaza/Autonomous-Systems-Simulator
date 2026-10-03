"""Phase 5 — PPO multi-environment edge cases: termination, truncation,
bootstrap, seed independence."""
import numpy as np
import pytest

from sim_client.agents.ppo_baseline import PPORunner


class ScriptedEnv:
    """Minimal gym-style env with scripted episode end."""

    def __init__(self, obs_dim=8, term_at=None, trunc_at=None, reward=1.0):
        self.term_at = term_at
        self.trunc_at = trunc_at
        self.reward = reward
        self.obs_dim = obs_dim
        self.t = 0
        self.reset_seeds = []

    def reset(self, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.t = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        term = self.term_at is not None and self.t >= self.term_at
        trunc = self.trunc_at is not None and self.t >= self.trunc_at
        info = {
            "step": self.t,
            "termination_reason": ("collision" if term
                                   else ("scenario_time_limit_exceeded" if trunc else "")),
            "lateral_offset": 0.0, "speed": 1.0,
            "checkpoints_passed": 0, "is_colliding": term, "is_on_road": True,
        }
        return (np.ones(self.obs_dim, dtype=np.float32) * self.t,
                self.reward, term, trunc, info)


def _runner(envs, seed=11, num_steps=16):
    return PPORunner(env=envs, num_steps=num_steps, num_epochs=1,
                     batch_size=8, seed=seed)


class TestTerminationEdgeCases:
    def test_one_env_terminates_early(self):
        envs = [ScriptedEnv(term_at=5), ScriptedEnv(trunc_at=1000)]
        r = _runner(envs)
        episodes = []
        r.on_update = lambda s: episodes.extend(s["episodes"])
        r.train(total_timesteps=16)
        assert any(e["env_idx"] == 0 and e["length"] == 5 for e in episodes)
        assert r.metrics["episodes_completed"] >= 1

    def test_simultaneous_termination(self):
        envs = [ScriptedEnv(term_at=3), ScriptedEnv(term_at=3)]
        r = _runner(envs)
        episodes = []
        r.on_update = lambda s: episodes.extend(s["episodes"])
        r.train(total_timesteps=16)
        env0_eps = [e for e in episodes if e["env_idx"] == 0]
        env1_eps = [e for e in episodes if e["env_idx"] == 1]
        assert env0_eps and env1_eps

    def test_async_episode_lengths(self):
        envs = [ScriptedEnv(term_at=3), ScriptedEnv(term_at=6)]
        r = _runner(envs)
        episodes = []
        r.on_update = lambda s: episodes.extend(s["episodes"])
        r.train(total_timesteps=16)
        lengths = sorted(e["length"] for e in episodes)
        assert 3 in lengths and 6 in lengths

    def test_final_rollout_mid_episode_bootstrap(self):
        """Rollout ending mid-episode must bootstrap V(next_obs), not crash."""
        env = ScriptedEnv(trunc_at=10**9)
        r = _runner([env], num_steps=8)
        out = r.train(total_timesteps=8)
        assert out["total_timesteps"] >= 8


class TestTruncationBootstrap:
    def test_truncated_step_stores_final_value_not_terminal(self):
        """Time-limit truncation must bootstrap V(final_obs): final_values_buf
        nonzero while terms_buf is 0 at the truncated step."""
        env = ScriptedEnv(trunc_at=4)
        r = _runner([env], num_steps=8)
        r.train(total_timesteps=8)
        # step index 3 (4th step, 0-based) is the truncation step
        assert r.terms_buf[3].item() == 0.0
        assert r.dones_buf[4].item() == 1.0      # next step enters new episode
        assert r.final_values_buf[3].item() != 0.0

    def test_terminated_step_no_bootstrap_value(self):
        env = ScriptedEnv(term_at=4)
        r = _runner([env], num_steps=8)
        r.train(total_timesteps=8)
        assert r.terms_buf[3].item() == 1.0
        assert r.final_values_buf[3].item() == 0.0


class TestSeedIndependence:
    def test_reset_seeds_differ_per_episode(self):
        env = ScriptedEnv(term_at=3)
        r = _runner([env], num_steps=8)
        r.train(total_timesteps=16)
        # every reset gets a distinct deterministic seed — never identical
        assert len(env.reset_seeds) >= 3
        assert len(set(env.reset_seeds)) == len(env.reset_seeds)

    def test_same_seed_reproducible(self):
        env_a, env_b = ScriptedEnv(term_at=4), ScriptedEnv(term_at=4)
        r1 = _runner([env_a], seed=5)
        r2 = _runner([env_b], seed=5)
        m1 = r1.train(total_timesteps=16)
        m2 = r2.train(total_timesteps=16)
        assert m1["episode_returns"] == m2["episode_returns"]

    def test_per_env_seed_streams_differ(self):
        envs = [ScriptedEnv(term_at=4), ScriptedEnv(term_at=4)]
        r = _runner(envs, num_steps=8)
        r.train(total_timesteps=16)
        assert set(envs[0].reset_seeds).isdisjoint(set(envs[1].reset_seeds))
