"""
Vectorized environment abstraction (Phase 6).

A VectorEnv presents N independent environments behind a batched
step/reset API so trainers can step all envs with one call. Backends
decide execution:

- SyncVectorEnv    — wraps a plain env list, sequential (Phase-5 semantics)
- ProcessVectorEnv — one worker process per env (see process_env.py)

Runners detect the interface via `is_vec_env` and take a vector rollout
path; plain lists keep the legacy sequential path unchanged.

Guarantees every backend must provide:

- stable per-env index (env i stays env i for the pool's lifetime)
- deterministic reset seeds (explicit seed argument per reset)
- isolated runtime state (no cross-env mutation)
- independent episode lifecycles (done on env i does not reset env j)
- correct (obs, reward, terminated, truncated, info) tuples per step
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

StepResult = Tuple[Any, float, bool, bool, Dict[str, Any]]


def is_vec_env(obj: Any) -> bool:
    """Duck-type check: batched step API marks an object as a VectorEnv."""
    return hasattr(obj, "step_all") and hasattr(obj, "reset_all")


class VectorEnv:
    """Interface documentation base — backends are duck-typed, not ABC-bound."""

    num_envs: int
    kind: str = "vector"

    def reset_all(self, seeds: List[Optional[int]]) -> List[Any]:
        raise NotImplementedError

    def reset_at(self, index: int, seed: Optional[int] = None) -> Any:
        raise NotImplementedError

    def step_all(self, actions: List[Any]) -> List[StepResult]:
        raise NotImplementedError

    def set_scenario(self, scenario_dict: Optional[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SyncVectorEnv(VectorEnv):
    """
    Synchronous in-process vector env — wraps a list of gym-style envs and
    loops over them in order. Semantically identical to Phase-5 sequential
    stepping; exists so all runners can share one vector code path.
    """

    kind = "sync"

    def __init__(self, envs: List[Any]):
        if not envs:
            raise ValueError("SyncVectorEnv requires at least one env")
        self.envs = list(envs)
        self.num_envs = len(self.envs)
        self._closed = False

    def reset_all(self, seeds: List[Optional[int]]) -> List[Any]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        return [env.reset(seed=s)[0] for env, s in zip(self.envs, seeds)]

    def reset_at(self, index: int, seed: Optional[int] = None) -> Any:
        return self.envs[index].reset(seed=seed)[0]

    def step_all(self, actions: List[Any]) -> List[StepResult]:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        return [env.step(a) for env, a in zip(self.envs, actions)]

    def set_scenario(self, scenario_dict: Optional[Dict[str, Any]]) -> None:
        from sim_env.scenario_designer import ScenarioDefinition
        scenario_def = (
            ScenarioDefinition.from_dict(scenario_dict)
            if isinstance(scenario_dict, dict) else scenario_dict
        )
        for env in self.envs:
            if not hasattr(env, "set_scenario"):
                raise NotImplementedError(
                    f"{type(env).__name__} does not support set_scenario"
                )
            env.set_scenario(scenario_def)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
