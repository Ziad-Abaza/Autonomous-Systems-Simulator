"""OffPolicyTrainer + OnPolicyTrainer — in-process env loop, metrics, ckpts.

The env loop is the single place that touches SimulationEnvironment:
- encoder.encode() runs once per step (cur frame reused next iter),
- mutator.draw_and_apply() installs a fresh scenario BEFORE each reset,
- step-after-done is structurally impossible (reset happens in-branch),
- episode metrics come only from info[] diagnostics — never oracle state.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np

from agentRL.checkpoints.io import save_checkpoint, write_run_state
from agentRL.core.agent import BaseRLAgent
from agentRL.core.config import TrainConfig
from agentRL.envs.factory import EnvFactory
from agentRL.envs.scenario_gen import ScenarioMutator
from agentRL.envs.track_registry import TrackSpec
from agentRL.obs.encoder import ObsEncoder
from agentRL.train.metrics import MetricsLogger

LOG_EVERY = 50  # heartbeat steps


class OffPolicyTrainer:
    """Single-env trainer for off-policy agents (SAC)."""

    def __init__(self, factory: EnvFactory, track: TrackSpec,
                 agent: BaseRLAgent, cfg: TrainConfig,
                 mutator: ScenarioMutator | None = None,
                 eval_fn: Callable[[BaseRLAgent], dict] | None = None):
        self.factory = factory
        self.track = track
        self.cfg = cfg
        self.mutator = mutator
        self.eval_fn = eval_fn
        self.env = factory.build(track, seed=cfg.seed)
        self.agent = agent
        if cfg.resume_from:
            self.agent = type(agent).load(cfg.resume_from)
        self.logger = MetricsLogger(cfg.run_dir)
        self.ckpt_dir = os.path.join(cfg.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._ep = {"return": 0.0, "len": 0, "speed": 0.0,
                    "steer_prev": 0.0, "steer_dsq": 0.0}

    # ------------------------------------------------------------- episodes

    def _new_episode(self) -> np.ndarray:
        if self.mutator is not None:
            self.mutator.draw_and_apply(self.env)
        obs, _ = self.env.reset()
        self.agent.encoder.reset()
        self.agent.adapter.reset()
        self._ep = {"return": 0.0, "len": 0, "speed": 0.0,
                    "steer_prev": 0.0, "steer_dsq": 0.0}
        return np.asarray(obs, dtype=np.float32)

    def _episode_metrics(self, info: dict[str, Any],
                         timestep: int) -> dict[str, Any]:
        ep = self._ep
        n = max(1, ep["len"])
        return {
            "return": ep["return"], "length": ep["len"],
            "mean_speed": ep["speed"] / n,
            "steer_smoothness": (ep["steer_dsq"] / n) ** 0.5,
            "termination_reason": info.get("termination_reason", ""),
            "lap_progress": info.get("lap_progress", 0.0),
            "laps_completed": info.get("laps_completed", 0),
            "checkpoints": info.get("checkpoints_passed", 0),
        }

    # -------------------------------------------------------------- training

    def train(self) -> dict[str, Any]:
        t0 = time.time()
        agent, cfg = self.agent, self.cfg
        enc = self.agent.encoder
        obs = self._new_episode()
        encoded = enc.encode(obs, None)
        ep_count = 0
        last_metrics: dict[str, Any] = {}

        for step in range(cfg.total_steps):
            a_env, aux = agent.act(encoded)
            # NOTE: step self.env each iteration — MixedTrackTrainer swaps
            # self.env on track changes; a captured local goes stale and
            # steps an already-done env forever.
            nobs, reward, terminated, truncated, info = self.env.step(a_env)
            if info.get("error"):
                # step-after-done class error — should be structurally
                # impossible; count it and force a fresh episode.
                self.logger.failure(step, {"error": info["error"]})
                obs = self._new_episode()
                encoded = enc.encode(obs, None)
                continue
            nenc = enc.encode(np.asarray(nobs, np.float32),
                              info.get("last_action", a_env))
            agent.observe(encoded, a_env, float(reward), nenc,
                          terminated, truncated, info)
            losses = agent.update(step)
            if losses:
                self.logger.update(step, losses)
                last_metrics.update(losses)

            ep = self._ep
            ep["return"] += float(reward)
            ep["len"] += 1
            ep["speed"] += float(info.get("speed", 0.0))
            steer = float(a_env[0])
            ep["steer_dsq"] += (steer - ep["steer_prev"]) ** 2
            ep["steer_prev"] = steer

            done = terminated or truncated
            if done:
                ep_count += 1
                self.logger.episode(step, self._episode_metrics(info, step))
                obs = self._new_episode()
                encoded = enc.encode(obs, None)
            else:
                encoded = nenc

            if step % LOG_EVERY == 0 or step == cfg.total_steps - 1:
                self.logger.heartbeat(step, {
                    "episodes": ep_count, "ep_return": ep["return"],
                    "ep_len": ep["len"],
                    "speed": info.get("speed", 0.0),
                    "buf": last_metrics.get("buf"),
                    "sps": (step + 1) / max(0.01, time.time() - t0),
                })
            if self.eval_fn and (step + 1) % cfg.eval_interval == 0:
                self.logger.eval(step, self.eval_fn(agent))
            if (step + 1) % cfg.ckpt_interval == 0:
                self._save(step)

        self._save(cfg.total_steps - 1)
        write_run_state(self.cfg.run_dir, {
            "timestep": cfg.total_steps, "episodes": ep_count,
            "wall_s": round(time.time() - t0, 2),
            "track": self.track.track_id,
            "algo": agent.algo_id,
            "train_state": dict(agent.train_state),
        })
        self.logger.close()
        return {"timesteps": cfg.total_steps, "episodes": ep_count,
                "run_dir": self.cfg.run_dir,
                "wall_s": round(time.time() - t0, 2)}

    def _save(self, step: int) -> None:
        save_checkpoint(self.agent,
                        os.path.join(self.ckpt_dir, "latest.pt"))
        save_checkpoint(self.agent,
                        os.path.join(self.ckpt_dir,
                                     f"step_{step + 1}.pt"))


class SyncVectorEnv:
    """N in-process envs stepped in lockstep, per-env encoders/mutators."""

    def __init__(self, factory: EnvFactory, track: TrackSpec, n: int,
                 obs_spec, seed: int, mutator: ScenarioMutator | None = None):
        self.envs = [factory.build(track, seed=seed + i) for i in range(n)]
        self.encoders = [ObsEncoder(obs_spec) for _ in range(n)]
        # per-env mutator forks — draws aren't coupled to step interleave
        self.mutators = ([mutator.fork(seed + i * 977) for i in range(n)]
                         if mutator is not None else None)

    def __len__(self) -> int:
        return len(self.envs)

    def reset_i(self, i: int) -> np.ndarray:
        if self.mutators is not None:
            self.mutators[i].draw_and_apply(self.envs[i])
        obs, _ = self.envs[i].reset()
        self.encoders[i].reset()
        return np.asarray(obs, dtype=np.float32)


class OnPolicyTrainer:
    """Vector-env trainer for on-policy agents (PPO)."""

    def __init__(self, factory: EnvFactory, track: TrackSpec,
                 agent: BaseRLAgent, cfg: TrainConfig,
                 mutator: ScenarioMutator | None = None,
                 eval_fn: Callable[[BaseRLAgent], dict] | None = None):
        self.factory = factory
        self.track = track
        self.cfg = cfg
        self.mutator = mutator
        self.eval_fn = eval_fn
        self.agent = agent
        if cfg.resume_from:
            self.agent = type(agent).load(cfg.resume_from)
        self.logger = MetricsLogger(cfg.run_dir)
        self.ckpt_dir = os.path.join(cfg.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # use the POST-resume agent's spec — a checkpoint may carry a
        # different observation layout than the template argument
        self.venv = SyncVectorEnv(factory, track, cfg.num_envs,
                                  self.agent.obs_spec, cfg.seed, mutator)

    def train(self) -> dict[str, Any]:
        t0 = time.time()
        agent, cfg = self.agent, self.cfg
        n = len(self.venv)
        obs = [self.venv.reset_i(i) for i in range(n)]
        enc = [self.venv.encoders[i].encode(o, None)
               for i, o in enumerate(obs)]
        eps = [{"return": 0.0, "len": 0, "speed": 0.0} for _ in range(n)]
        ep_count = 0

        for step in range(cfg.total_steps):
            i = step % n
            env = self.venv.envs[i]
            a_env, aux = agent.act(enc[i])
            nobs, reward, terminated, truncated, info = env.step(a_env)
            nenc = self.venv.encoders[i].encode(
                np.asarray(nobs, np.float32),
                info.get("last_action", a_env))
            agent.observe(enc[i], a_env, float(reward), nenc,
                          terminated, truncated, {**aux, "env_id": i})
            eps[i]["return"] += float(reward)
            eps[i]["len"] += 1
            eps[i]["speed"] += float(info.get("speed", 0.0))
            losses = agent.update(step)
            if losses:
                self.logger.update(step, losses)

            done = terminated or truncated
            if done:
                ep_count += 1
                ep = eps[i]
                self.logger.episode(step, {
                    "return": ep["return"], "length": ep["len"],
                    "mean_speed": ep["speed"] / max(1, ep["len"]),
                    "termination_reason": info.get("termination_reason", ""),
                    "lap_progress": info.get("lap_progress", 0.0),
                    "env_id": i,
                })
                eps[i] = {"return": 0.0, "len": 0, "speed": 0.0}
                enc[i] = self.venv.encoders[i].encode(
                    self.venv.reset_i(i), None)
            else:
                enc[i] = nenc

            if step % LOG_EVERY == 0 or step == cfg.total_steps - 1:
                self.logger.heartbeat(step, {
                    "episodes": ep_count,
                    "sps": (step + 1) / max(0.01, time.time() - t0)})
            if self.eval_fn and (step + 1) % cfg.eval_interval == 0:
                self.logger.eval(step, self.eval_fn(agent))
            if (step + 1) % cfg.ckpt_interval == 0:
                save_checkpoint(self.agent, os.path.join(
                    self.ckpt_dir, "latest.pt"))

        save_checkpoint(self.agent, os.path.join(
            self.ckpt_dir, "latest.pt"))
        write_run_state(self.cfg.run_dir, {
            "timestep": cfg.total_steps, "episodes": ep_count,
            "wall_s": round(time.time() - t0, 2),
            "track": self.track.track_id, "algo": agent.algo_id,
            "train_state": dict(agent.train_state)})
        self.logger.close()
        return {"timesteps": cfg.total_steps, "episodes": ep_count,
                "run_dir": self.cfg.run_dir,
                "wall_s": round(time.time() - t0, 2)}


class TrackSampler:
    """Multi-track env pool: samples a cached env per episode.

    Env instances are built once per track and reused — the sampler only
    decides which track the next episode runs on. The rehearsal buffer
    rotates via memory.begin_track(track_id) on each switch.
    """

    def __init__(self, factory: EnvFactory, tracks: list[TrackSpec],
                 seed: int, mutator: ScenarioMutator | None = None):
        self.tracks = list(tracks)
        self.envs = {t.track_id: factory.build(t, seed=seed + i * 101)
                     for i, t in enumerate(tracks)}
        for t in self.tracks:
            setattr(self.envs[t.track_id], "_sampler_track", t.track_id)
        # per-track mutator forks — each track's scenario stream is
        # independent of which track was sampled before it
        self.mutators = ({t.track_id: mutator.fork(seed + i * 997)
                          for i, t in enumerate(tracks)}
                         if mutator is not None else None)
        self._rng = np.random.default_rng(seed)

    def next(self):
        t = self.tracks[int(self._rng.integers(len(self.tracks)))]
        return self.envs[t.track_id]


class MixedTrackTrainer(OffPolicyTrainer):
    """Off-policy trainer that samples a track per episode (E004)."""

    def __init__(self, factory: EnvFactory, tracks: list[TrackSpec],
                 agent: BaseRLAgent, cfg: TrainConfig,
                 mutator: ScenarioMutator | None = None,
                 eval_fn=None):
        self.sampler = TrackSampler(factory, tracks, cfg.seed,
                                    mutator=mutator)
        super().__init__(factory, tracks[0], agent, cfg, mutator, eval_fn)

    def _new_episode(self) -> np.ndarray:
        self.env = self.sampler.next()
        mem = getattr(self.agent, "memory", None)
        if mem is not None and hasattr(mem, "begin_track"):
            mem.begin_track(self.env._sampler_track)
        if self.sampler.mutators is not None:
            self.sampler.mutators[self.env._sampler_track] \
                .draw_and_apply(self.env)
        obs, _ = self.env.reset()
        self.agent.encoder.reset()
        self.agent.adapter.reset()
        self._ep = {"return": 0.0, "len": 0, "speed": 0.0,
                    "steer_prev": 0.0, "steer_dsq": 0.0}
        return np.asarray(obs, dtype=np.float32)

    def _episode_metrics(self, info, timestep):
        m = super()._episode_metrics(info, timestep)
        m["track"] = getattr(self.env, "_sampler_track",
                             self.track.track_id)
        return m


def make_trainer(factory: EnvFactory, track: TrackSpec,
                 agent: BaseRLAgent, cfg: TrainConfig,
                 mutator: ScenarioMutator | None = None,
                 eval_fn=None):
    if agent.is_on_policy:
        return OnPolicyTrainer(factory, track, agent, cfg, mutator, eval_fn)
    return OffPolicyTrainer(factory, track, agent, cfg, mutator, eval_fn)
