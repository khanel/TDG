"""
Generalized reward computation for RL environment.
Problem-agnostic, based on solver metrics and phase transitions.
"""

from typing import Optional
import numpy as np


class RewardComputer:
    """Computes rewards from solver state and actions."""

    def __init__(self, problem_bounds: dict, *, clip_range: tuple[float, float] = (-1.0, 1.0)):
        self.lower_bound = problem_bounds.get("lower_bound", 0.0)
        self.upper_bound = problem_bounds.get("upper_bound", 1.0)
        lo, hi = clip_range
        if lo > hi:
            lo, hi = hi, lo
        self._clip_min = float(lo)
        self._clip_max = float(hi)

    def compute(self, action: int, phase: str, improvement: float, terminated: bool) -> float:
        """Compute reward based on action and state."""
        reward = 0.0
        if improvement > 0:
            reward += improvement / max(1e-9, self.upper_bound - self.lower_bound)
        if action == 1 and phase == "exploration":
            reward += 0.1  # Bonus for switching
        if terminated:
            reward += 0.5  # Bonus for termination
        return float(np.clip(reward, self._clip_min, self._clip_max))
