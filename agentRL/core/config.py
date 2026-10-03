"""Configuration dataclasses for agents and training runs."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentConfig:
    algo_id: str
    hidden_sizes: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        d = dict(data)
        d["hidden_sizes"] = tuple(d.get("hidden_sizes", (256, 256)))
        return cls(**d)


@dataclass
class TrainConfig:
    total_steps: int
    eval_interval: int = 10_000
    ckpt_interval: int = 10_000
    num_envs: int = 1
    seed: int = 42
    run_dir: str = "agentRL/runs/default"
    eval_episodes: int = 3
    resume_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
