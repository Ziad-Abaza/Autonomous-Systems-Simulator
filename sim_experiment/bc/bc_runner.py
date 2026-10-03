"""
BCRunner — trains a BCPolicy with supervised regression/classification.

continuous action_mode: MSE on action vectors
discrete  action_mode : cross-entropy on integer action indices

Deterministic: fixed torch/np seeds, index-order shuffling via seeded
permutation (not DataLoader worker RNG).
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sim_experiment.bc.bc_model import BCPolicy


class BCRunner:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_mode: str = "continuous",
        hidden: Tuple[int, ...] = (64, 64),
        lr: float = 3e-4,
        batch_size: int = 256,
        seed: int = 42,
        device: str = "cpu",
        resume_checkpoint: Optional[str] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.action_mode = action_mode
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.device = torch.device(device)
        self.epoch_offset = 0

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self.hidden = tuple(int(h) for h in hidden)
        self.policy = BCPolicy(obs_dim, act_dim, self.hidden, action_mode).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        if action_mode == "continuous":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

    # ------------------------------------------------------------- training

    def _prep(self, data: Dict[str, np.ndarray]):
        obs = torch.tensor(np.asarray(data["obs"]), dtype=torch.float32,
                           device=self.device)
        if self.action_mode == "continuous":
            act = torch.tensor(np.asarray(data["actions"], dtype=np.float32),
                               dtype=torch.float32, device=self.device)
        else:
            act = torch.tensor(np.asarray(data["actions"]).reshape(-1),
                               dtype=torch.long, device=self.device)
        return obs, act

    def _loss(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        out = self.policy(obs)
        return self.loss_fn(out, act)

    def train_epochs(
        self,
        train: Dict[str, np.ndarray],
        val: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 10,
        on_epoch=None,
    ) -> Dict[str, Any]:
        """Trains `epochs` full passes; returns training metrics."""
        obs, act = self._prep(train)
        n = obs.shape[0]
        val_tensors = self._prep(val) if val is not None else None

        losses = []
        val_losses = []
        initial_loss = None

        for epoch in range(1, epochs + 1):
            perm = self._rng.permutation(n)
            epoch_losses = []
            for i in range(0, n, self.batch_size):
                idx = torch.tensor(perm[i:i + self.batch_size],
                                   dtype=torch.long, device=self.device)
                loss = self._loss(obs[idx], act[idx])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                epoch_losses.append(float(loss.item()))

            mean_loss = float(np.mean(epoch_losses))
            if initial_loss is None:
                initial_loss = mean_loss
            losses.append(mean_loss)

            val_loss = None
            if val_tensors is not None:
                with torch.no_grad():
                    vobs, vact = val_tensors
                    val_loss = float(self._loss(vobs, vact).item())
                val_losses.append(val_loss)

            if on_epoch is not None:
                on_epoch({
                    "epoch": self.epoch_offset + epoch,
                    "train_loss": mean_loss,
                    "val_loss": val_loss,
                })

        self.epoch_offset += epochs
        return {
            "epochs_trained": epochs,
            "epoch_losses": losses,
            "val_losses": val_losses,
            "initial_train_loss": initial_loss if initial_loss is not None else 0.0,
            "final_train_loss": losses[-1] if losses else 0.0,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "transitions": int(n),
        }

    # ------------------------------------------------------------- eval/checkpoint

    def eval_action_fn(self):
        """Deterministic policy callable for evaluation loops."""
        def act(obs):
            if isinstance(obs, dict):
                obs = obs.get("vector", np.zeros(self.obs_dim, dtype=np.float32))
            x = torch.tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1),
                             dtype=torch.float32, device=self.device)
            with torch.no_grad():
                out = self.policy(x).squeeze(0)
            if self.action_mode == "continuous":
                return out.cpu().numpy()
            return int(out.argmax().item())
        return act

    def save_checkpoint(self, path: str, epoch: int = 0,
                        extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "algorithm": "bc",
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "action_mode": self.action_mode,
            "hidden": list(self.hidden),
            "epoch": int(epoch),
            "seed": self.seed,
        }
        payload.update(dict(extra or {}))
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch_offset = int(ckpt.get("epoch", 0))
        return ckpt
