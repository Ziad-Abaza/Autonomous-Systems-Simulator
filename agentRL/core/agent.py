"""BaseRLAgent — the stable extension point for all agentRL algorithms.

Contract:
- Policies emit normalized actions in [-1,1]^act_dim; `act()` returns
  env-space actions via the shared ActionAdapter. `observe()` receives
  the env-space action actually sent to the sim.
- obs passed to act/observe is the ENCODED observation (encoder output =
  ObservationSpec layout), so agents never care about frame stacking or
  prev_action — those only change input_dim.
- done semantics: terminated & not truncated -> absorbing; truncated ->
  bootstrap. Both-true (step-after-done) counts as truncation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from agentRL.act.adapter import ActionAdapter
from agentRL.core.config import AgentConfig
from agentRL.obs.encoder import ObsEncoder
from agentRL.obs.spec import ObservationSpec


class BaseRLAgent(ABC):
    algo_id: str = "base"
    is_on_policy: bool = False

    def __init__(self, obs_spec: ObservationSpec,
                 act_space: dict[str, np.ndarray], cfg: AgentConfig,
                 encoder: ObsEncoder | None = None):
        self.obs_spec = obs_spec
        self.act_space = {"low": np.asarray(act_space["low"], np.float32),
                          "high": np.asarray(act_space["high"], np.float32)}
        self.cfg = cfg
        self.adapter = ActionAdapter(self.act_space["low"],
                                     self.act_space["high"])
        self.encoder = encoder or ObsEncoder(obs_spec)
        self.memory: Any = None
        self.train_state: dict[str, int | float] = {
            "steps": 0, "updates": 0, "nan_guards": 0, "episodes": 0,
        }

    # -- action-space helpers ------------------------------------------------

    def act_space_to_env(self, raw: np.ndarray) -> np.ndarray:
        return self.adapter.to_env(np.asarray(raw, dtype=np.float32))

    def env_to_act_space(self, env_action: np.ndarray) -> np.ndarray:
        return self.adapter.from_env(env_action)

    # -- interface -----------------------------------------------------------

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False
            ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return (env-space action, aux info)."""

    @abstractmethod
    def observe(self, obs: np.ndarray, action: np.ndarray, reward: float,
                next_obs: np.ndarray, terminated: bool, truncated: bool,
                info: dict[str, Any]) -> None:
        """Record one transition (env-space action, encoded obs)."""

    @abstractmethod
    def update(self, step: int) -> dict[str, float]:
        """Run learning updates; returns metrics (may be empty)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Write a schema-versioned checkpoint."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseRLAgent":
        """Load a checkpoint; refuses on version/spec mismatch."""
