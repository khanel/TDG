"""
Minimal reward computation for the RL environment.
"""

import logging
import math
from typing import Optional
import numpy as np

class RewardComputer:
    """
    Computes rewards based on a minimal set of signals:
    - Improvement in solution quality.
    - Penalties/rewards for taking actions (STAY vs. ADVANCE).
    """

    def __init__(
        self,
        problem_meta: dict,
        *,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        time_penalty: float = -0.01,
        stagnation_threshold: float = 0.8,
        advance_reward: float = 0.5,
        advance_penalty: float = -0.5,
        logger: logging.Logger,
    ):
        self.logger = logger
        self.lower_bound, self.upper_bound = self._extract_bounds(problem_meta)
        self.fitness_range = max(1e-9, self.upper_bound - self.lower_bound)
        
        self._clip_min, self._clip_max = sorted(clip_range)
        self.time_penalty = float(time_penalty)
        self.stagnation_threshold = float(stagnation_threshold)
        self.advance_reward = float(advance_reward)
        self.advance_penalty = float(advance_penalty)

        self.logger.debug("RewardComputer (minimal) initialized.")

    def compute(
        self,
        *,
        action: int,
        improvement: float,
        observation: np.ndarray,
        **kwargs, # Absorb unused arguments from the environment
    ) -> float:
        """
        Computes the reward for a given step.
        """
        # 1. Improvement Reward
        normalized_improvement = float(np.clip(improvement / self.fitness_range, -1.0, 1.0))
        
        # 2. Action Reward/Penalty
        action_reward = 0.0
        if action == 0:  # STAY
            action_reward = self.time_penalty
        elif action == 1:  # ADVANCE
            stagnation_level = observation[3] # Index 3 is 'stagnation'
            if stagnation_level >= self.stagnation_threshold:
                action_reward = self.advance_reward # Reward for advancing from stagnation
            else:
                action_reward = self.advance_penalty # Penalize for advancing from a productive state

        total_reward = normalized_improvement + action_reward
        final_reward = float(np.clip(total_reward, self._clip_min, self._clip_max))
        
        self.logger.debug(
            f"Reward: total={total_reward:.4f}, final={final_reward:.4f} "
            f"(improvement={normalized_improvement:.4f}, action_reward={action_reward:.4f})"
        )
        return final_reward

    @staticmethod
    def _extract_bounds(meta: dict) -> tuple[float, float]:
        """Extracts fitness bounds from problem metadata."""
        if not isinstance(meta, dict):
            return 0.0, 1.0
        
        keys_to_try = [
            ("lower_bound", "upper_bound"),
            ("fitness_lower_bound", "fitness_upper_bound"),
            ("fitness_min", "fitness_max"),
        ]
        
        for lo_key, hi_key in keys_to_try:
            if lo_key in meta and hi_key in meta:
                try:
                    lb = float(meta[lo_key])
                    ub = float(meta[hi_key])
                    if math.isfinite(lb) and math.isfinite(ub) and ub > lb:
                        return lb, ub
                except (ValueError, TypeError):
                    continue
                    
        return 0.0, 1.0