"""EvalMatrix — checkpoint x track evaluation grid -> eval_matrix.json."""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

from agentRL.envs.factory import EnvFactory
from agentRL.envs.scenario_gen import ScenarioMutator
from agentRL.envs.track_registry import TrackSpec
from agentRL.eval.evaluate import evaluate_policy


class EvalMatrix:
    """Runs each policy on each track; cells hold aggregate metrics."""

    def __init__(self, factory: EnvFactory):
        self.factory = factory

    def run(
        self,
        policies: dict[str, Any],          # name -> agent (BaseRLAgent)
        tracks: list[TrackSpec],
        out_dir: str | None = None,
        seeds: Iterable[int] = (42, 43, 44, 45, 46),
        mutator: ScenarioMutator | None = None,
        max_steps: int = 2000,
    ) -> dict[str, dict[str, dict]]:
        failure_rows: list[dict] = []
        grid: dict[str, dict[str, dict]] = {}
        for name, agent in policies.items():
            grid[name] = {}
            for track in tracks:
                res = evaluate_policy(
                    self.factory, track, agent, seeds=seeds,
                    mutator=mutator, max_steps=max_steps,
                    out_rows=failure_rows)
                for row in res["per_episode"]:
                    row["policy"] = name
                grid[name][track.track_id] = res["aggregate"]

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "eval_matrix.json"), "w",
                      encoding="utf-8") as f:
                json.dump(grid, f, indent=2)
            with open(os.path.join(out_dir, "failures.jsonl"), "w",
                      encoding="utf-8") as f:
                for row in failure_rows:
                    if row["failure_class"] != "completed":
                        f.write(json.dumps(row, default=str) + "\n")
        return grid
