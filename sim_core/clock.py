"""
Fixed-timestep clock and deterministic simulation timing.
Supports decoupled physics frequency, variable rendering tick,
seedable pseudo-random number generator, and deterministic replay stepping.
"""

from __future__ import annotations
import time
import numpy as np


class FixedClock:
    """
    Manages deterministic time progression with a fixed physics timestep.
    """
    def __init__(self, physics_hz: float = 60.0, seed: int = 42):
        self.physics_hz = float(physics_hz)
        self.dt = 1.0 / self.physics_hz
        self.step_count: int = 0
        self.sim_time: float = 0.0
        self.time_scale: float = 1.0
        self.is_paused: bool = False

        # Deterministic RNG
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Real-time synchronization
        self._last_real_time = time.perf_counter()
        self._accumulator = 0.0

    def reset(self, seed: int | None = None) -> None:
        """Resets the clock back to t=0, re-seeding if requested."""
        self.step_count = 0
        self.sim_time = 0.0
        self._accumulator = 0.0
        self._last_real_time = time.perf_counter()
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

    def advance_fixed_step(self) -> float:
        """
        Advances the simulation by exactly one fixed physics timestep.
        Returns the dt for this step.
        """
        self.step_count += 1
        self.sim_time += self.dt
        return self.dt

    def compute_steps_for_real_frame(self, max_sub_steps: int = 5) -> int:
        """
        Computes how many physics sub-steps should be executed to catch up
        with real elapsed time, preventing spiral of death.
        """
        if self.is_paused:
            self._last_real_time = time.perf_counter()
            return 0

        now = time.perf_counter()
        elapsed = (now - self._last_real_time) * self.time_scale
        self._last_real_time = now

        # Prevent spiral-of-death on slow frames (limit to max 0.2s elapsed)
        elapsed = min(elapsed, 0.2)
        self._accumulator += elapsed

        steps = 0
        while self._accumulator >= self.dt and steps < max_sub_steps:
            self._accumulator -= self.dt
            steps += 1

        return steps
