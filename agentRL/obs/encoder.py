"""ObsEncoder — env flat-vector -> policy input tensor layout.

Pure numpy. Maintains a frame deque for stacking and appends the
previous action when the spec asks for it. The environment's flat
vector already equals spec.channel_names order (EnvFactory builds the
channel subset), so no slicing happens here — only stacking/concat.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from agentRL.obs.spec import ACTION_DIM, ObservationSpec


class ObsEncoder:
    def __init__(self, spec: ObservationSpec):
        self.spec = spec
        self._frames: deque[np.ndarray] = deque(maxlen=spec.frame_stack)

    def reset(self) -> np.ndarray:
        self._frames.clear()
        return np.zeros(self.spec.input_dim, dtype=np.float32)

    def encode(self, obs_flat: np.ndarray,
               prev_action: np.ndarray | None = None) -> np.ndarray:
        if self.spec.image:
            raise NotImplementedError(
                "image observations are reserved in ObservationSpec; "
                "CNN support lands post-E009 (see AGENT_RL_ARCHITECTURE.md)"
            )
        obs_flat = np.asarray(obs_flat, dtype=np.float32).ravel()
        if obs_flat.size != self.spec.vector_dim:
            raise ValueError(
                f"obs vector dim {obs_flat.size} != spec vector_dim "
                f"{self.spec.vector_dim}"
            )
        self._frames.append(obs_flat.copy())
        while len(self._frames) < self.spec.frame_stack:
            self._frames.appendleft(obs_flat.copy())
        parts = [np.concatenate(list(self._frames))]
        if self.spec.prev_action:
            pa = (np.zeros(ACTION_DIM, dtype=np.float32) if prev_action is None
                  else np.asarray(prev_action, dtype=np.float32).ravel())
            if pa.size != ACTION_DIM:
                raise ValueError(f"prev_action dim {pa.size} != {ACTION_DIM}")
            parts.append(pa)
        return np.concatenate(parts).astype(np.float32)
