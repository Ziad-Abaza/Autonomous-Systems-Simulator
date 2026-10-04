"""ContinualTrainer — sequential multi-track phases + retention eval.

Each phase trains the SAME agent on a new track. After each phase the
eval matrix re-measures every previously-seen track + holdouts, so the
report captures retention/forgetting directly (not just final scores).

Resume: `continual_state.json` persists completed phase count, last
checkpoint, and accumulated eval cells — a killed sequence restarts
mid-run with prior evidence intact.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from agentRL.checkpoints.io import save_checkpoint
from agentRL.core.agent import BaseRLAgent
from agentRL.core.config import TrainConfig
from agentRL.envs.factory import EnvFactory
from agentRL.envs.scenario_gen import ScenarioMutator
from agentRL.envs.track_registry import TrackRegistry
from agentRL.eval.matrix import EvalMatrix
from agentRL.train.trainer import make_trainer

_STATE_FILE = "continual_state.json"
_REPORT_FILE = "continual_report.json"


@dataclass(frozen=True)
class Phase:
    track_id: str
    steps: int
    mutator: ScenarioMutator | None = None
    tag: str = ""
    demos: int = 0  # PD warmstart steps seeded into this track's buffer
    train_kwargs: dict[str, Any] = field(default_factory=dict)


def _derive_metrics(cells: dict[str, dict[str, dict]],
                    trained_on: dict[str, int]) -> dict[str, Any]:
    """forgetting / retention / transfer from a phase x track cell grid."""
    phases = sorted(cells.keys())
    final = phases[-1]
    forgetting: dict[str, float] = {}
    transfer: dict[str, float] = {}
    retention_final: dict[str, float] = {}
    for tid in {t for cell in cells.values() for t in cell}:
        hist = [(p, cells[p][tid]["mean_return"]) for p in phases
                if tid in cells[p]]
        if not hist:
            continue
        retention_final[tid] = cells[final][tid]["mean_return"]
        trained_idx = trained_on.get(tid)
        if trained_idx is not None:
            # best performance at-or-after the track's own training phase:
            # pre-training evals are a transfer baseline, not retained skill
            post = [v for p, v in hist
                    if int(p.split("_")[1]) >= trained_idx]
            forgetting[tid] = cells[final][tid]["mean_return"] \
                - (max(post) if post else hist[-1][1])
            if trained_idx > 0:
                prev_key = f"phase_{trained_idx - 1}"
                post_key = f"phase_{trained_idx}"
                if prev_key in cells and tid in cells[prev_key] \
                        and post_key in cells and tid in cells[post_key]:
                    transfer[tid] = cells[post_key][tid]["mean_return"] \
                        - cells[prev_key][tid]["mean_return"]
        else:
            forgetting[tid] = 0.0
    return {"forgetting": forgetting, "retention_final": retention_final,
            "transfer": transfer}


class ContinualTrainer:
    def __init__(self, factory: EnvFactory, agent: BaseRLAgent,
                 cfg: TrainConfig, holdout_tracks: list[str] | None = None,
                 eval_seeds: tuple[int, ...] = (42, 43, 44),
                 eval_max_steps: int = 2000,
                 registry: TrackRegistry | None = None):
        self.factory = factory
        self.agent = agent
        self.cfg = cfg
        self.holdouts = holdout_tracks or []
        self.eval_seeds = eval_seeds
        self.eval_max_steps = eval_max_steps
        self.registry = registry or TrackRegistry.default()
        self.matrix = EvalMatrix(factory)
        os.makedirs(os.path.join(cfg.run_dir, "checkpoints"),
                    exist_ok=True)

    # ------------------------------------------------------------- run loop

    def run(self, phases: list[Phase], resume: bool = False
            ) -> dict[str, Any]:
        state = self._load_state() if resume else \
            {"completed": 0, "cells": {}, "last_ckpt": None}
        start = int(state.get("completed", 0))
        if resume and state.get("last_ckpt"):
            # mid-run restart must reload the learned weights, not just
            # skip completed phases — otherwise later phases train a
            # fresh agent and eval cells compare against nothing
            self.agent = type(self.agent).load(state["last_ckpt"])
        cells: dict[str, dict[str, dict]] = dict(state.get("cells", {}))

        for i, phase in enumerate(phases):
            if i < start:
                continue
            track = self.registry.load(phase.track_id)
            if getattr(self.agent, "memory", None) is not None and \
                    hasattr(self.agent.memory, "begin_track"):
                self.agent.memory.begin_track(track.track_id)
            sub_cfg = TrainConfig(
                total_steps=phase.steps,
                eval_interval=self.cfg.eval_interval,
                ckpt_interval=self.cfg.ckpt_interval,
                num_envs=self.cfg.num_envs, seed=self.cfg.seed + i,
                run_dir=os.path.join(self.cfg.run_dir,
                                     f"phase_{i}_{phase.track_id}"),
                eval_episodes=self.cfg.eval_episodes,
                extra=self.cfg.extra)
            trainer = make_trainer(self.factory, track, self.agent,
                                   sub_cfg, mutator=phase.mutator)
            if phase.demos > 0 and \
                    getattr(self.agent, "memory", None) is not None:
                from agentRL.baselines.pd_driver import collect_demos
                collect_demos(trainer.env, self.agent.adapter,
                              self.agent.memory, self.agent.encoder,
                              n_steps=phase.demos, mutator=phase.mutator)
            trainer.train()
            ckpt_path = os.path.join(
                self.cfg.run_dir, "checkpoints",
                f"phase_{i}_{phase.track_id}.pt")
            save_checkpoint(self.agent, ckpt_path)

            # eval ALL declared tracks every phase — future tracks get a
            # pre-training baseline so `transfer` is measured, not assumed
            all_tracks = [p.track_id for p in phases]
            eval_tracks = [self.registry.load(t) for t in
                           dict.fromkeys(all_tracks + self.holdouts)]
            cells[f"phase_{i}"] = self.matrix.run(
                {self.agent.algo_id: self.agent}, eval_tracks,
                seeds=self.eval_seeds, max_steps=self.eval_max_steps,
            )[self.agent.algo_id]
            self._save_state(i + 1, cells, ckpt_path, phases)
            self._write_report(phases, cells)

        return self._write_report(phases, cells)

    # ---------------------------------------------------------- persistence

    def _state_path(self) -> str:
        return os.path.join(self.cfg.run_dir, _STATE_FILE)

    def _load_state(self) -> dict[str, Any]:
        p = self._state_path()
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {"completed": 0, "cells": {}, "last_ckpt": None}

    def _save_state(self, completed: int, cells: dict, ckpt: str,
                    phases: list[Phase]) -> None:
        with open(self._state_path(), "w", encoding="utf-8") as f:
            json.dump({"completed": completed, "cells": cells,
                       "last_ckpt": ckpt,
                       "phases": [p.track_id for p in phases]}, f, indent=2)

    def _write_report(self, phases: list[Phase],
                      cells: dict) -> dict[str, Any]:
        trained_on = {p.track_id: i for i, p in enumerate(phases)}
        mem = getattr(self.agent, "memory", None)
        report = {
            "phases": [{"i": i, "track_id": p.track_id, "steps": p.steps,
                        "tag": p.tag} for i, p in enumerate(phases)],
            "holdouts": self.holdouts,
            "cells": cells,
            "derived": _derive_metrics(cells, trained_on) if cells else {},
            "rehearsal": mem.state_dict() if mem and
                       hasattr(mem, "state_dict") else {},
            "agent": {"algo_id": self.agent.algo_id,
                      "train_state": dict(self.agent.train_state)},
        }
        with open(os.path.join(self.cfg.run_dir, _REPORT_FILE), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return report
