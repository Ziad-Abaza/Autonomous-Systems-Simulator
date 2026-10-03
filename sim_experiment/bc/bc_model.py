"""Behavior Cloning policy network — plain MLP, deterministic forward."""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """
    obs -> hidden MLP -> action head.
    continuous: outputs raw means (squashed downstream as needed)
    discrete  : outputs per-class logits
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: Tuple[int, ...] = (64, 64),
                 action_mode: str = "continuous"):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.action_mode = action_mode
        layers = []
        d = self.obs_dim
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.Tanh()]
            d = int(h)
        layers.append(nn.Linear(d, self.act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
