"""
Generalized reward computation for RL environment.
Problem-agnostic, based on solver metrics and phase transitions.
This version implements a dynamic, multi-objective reward framework.
"""

import logging
from typing import Optional
import numpy as np


class RewardComputer:
    """
    Computes rewards using a dynamic, multi-objective framework.
    It balances solution quality (progress) and diversity (exploration)
    based on the remaining budget, and provides bonuses for strategic decisions.
    """

    def __init__(
        self,
        problem_bounds: dict,
        *,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        efficiency_penalty: float = 0.01,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the reward computer.
        Args:
            problem_bounds: Dictionary with fitness bounds ('lower_bound', 'upper_bound').
            clip_range: Tuple to clip the final reward.
            efficiency_penalty: Small constant penalty per step to encourage efficiency.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.lower_bound, self.upper_bound = self._extract_bounds(problem_bounds)
        lo, hi = clip_range
        if lo > hi:
            lo, hi = hi, lo
        self._clip_min = float(lo)
        self._clip_max = float(hi)
        self._fitness_range = max(1e-9, self.upper_bound - self.lower_bound)
        self.efficiency_penalty = float(efficiency_penalty)

        self.logger.info(f"RewardComputer initialized with:")
        self.logger.info(f"  problem_bounds: {problem_bounds}")
        self.logger.info(f"  clip_range: {clip_range}")
        self.logger.info(f"  efficiency_penalty: {self.efficiency_penalty}")

    def compute(
        self,
        *,
        action: int,
        phase: str,
        improvement: float,
        terminated: bool,
        observation: np.ndarray,
        prev_observation: Optional[np.ndarray] = None,
        steps_run: int = 0,
        switched: bool = False,
        phase_after: Optional[str] = None,
    ) -> float:
        """
        Compute reward based on the dynamic, multi-objective framework.
        The total reward combines progress, exploration, efficiency, and strategic decision bonuses,
        with weights dynamically adjusted based on the remaining budget.
        """
        obs = np.asarray(observation, dtype=float)
        prev_obs = np.asarray(prev_observation, dtype=float) if prev_observation is not None else obs

        # --- 1. Progress Reward (R_progress) ---
        # Normalized fitness delta. Positive for improvement.
        normalized_improvement = float(np.clip(improvement / self._fitness_range, -1.0, 1.0))
        progress_reward = normalized_improvement

        # --- 2. Intrinsic Exploration Reward (R_exploration) ---
        # Change in population diversity. Lower concentration (obs[4]) is better for exploration.
        current_concentration = obs[4]
        previous_concentration = prev_obs[4]
        # Reward is higher if concentration decreases (more exploration).
        exploration_reward = previous_concentration - current_concentration

        # --- 3. Strategic Decision Bonus (R_decision) ---
        decision_reward = 0.0
        if action == 1:  # Agent chose to switch or terminate
            if phase == "exploration" and switched:
                # Switch bonus: rewards switching from a good, diverse state (low concentration).
                norm_best_at_switch = prev_obs[1]
                concentration_at_switch = prev_obs[4]
                decision_reward = (1.0 - norm_best_at_switch) * (1.0 - concentration_at_switch)
            elif terminated:
                # Termination bonus: rewards finishing with a high-quality solution.
                final_norm_best = obs[1]
                decision_reward = 1.0 - final_norm_best
                # Penalty for stopping prematurely with significant budget left.
                budget_remaining = obs[0]
                if budget_remaining > 0.1:  # e.g., if more than 10% of budget is left
                    decision_reward -= budget_remaining

        # --- 4. Budget-Aware Weighting ---
        # B = budget_remaining, from 1 (start) to 0 (end).
        budget_remaining = obs[0]
        budget_used_ratio = 1.0 - budget_remaining

        # w_quality increases as budget is used, w_explore decreases.
        w_quality = budget_used_ratio  # (1 - B)
        w_explore = budget_remaining   # B

        # --- 5. Total Reward Combination ---
        # R_total = (w_quality * R_progress) + (w_explore * R_exploration) - C + R_decision
        # C is the efficiency penalty.
        total_reward = (
            (w_quality * progress_reward)
            + (w_explore * exploration_reward)
            - self.efficiency_penalty
            + decision_reward
        )

        final_reward = float(np.clip(total_reward, self._clip_min, self._clip_max))

        self.logger.info(f"Reward calculation:")
        self.logger.info(f"  - improvement: {improvement:.4f}, normalized: {normalized_improvement:.4f}")
        self.logger.info(f"  - progress_reward: {progress_reward:.4f}")
        self.logger.info(f"  - current_concentration: {current_concentration:.4f}, previous_concentration: {previous_concentration:.4f}")
        self.logger.info(f"  - exploration_reward: {exploration_reward:.4f}")
        self.logger.info(f"  - decision_reward: {decision_reward:.4f} (action={action}, phase={phase}, switched={switched}, terminated={terminated})")
        self.logger.info(f"  - budget_remaining: {budget_remaining:.4f}, w_quality: {w_quality:.4f}, w_explore: {w_explore:.4f}")
        self.logger.info(f"  - efficiency_penalty: {self.efficiency_penalty:.4f}")
        self.logger.info(f"  - total_reward (before clip): {total_reward:.4f}")
        self.logger.info(f"  - final_reward (after clip): {final_reward:.4f}")

        return final_reward

    @staticmethod
    def _extract_bounds(meta: dict) -> tuple[float, float]:
        if not isinstance(meta, dict):
            return 0.0, 1.0
        keys = [
            ("lower_bound", "upper_bound"),
            ("fitness_lower_bound", "fitness_upper_bound"),
            ("fitness_min", "fitness_max"),
        ]
        for lo_key, hi_key in keys:
            if lo_key in meta and hi_key in meta:
                try:
                    lb = float(meta[lo_key])
                    ub = float(meta[hi_key])
                except Exception:
                    continue
                if np.isfinite(lb) and np.isfinite(ub) and ub > lb:
                    return lb, ub
        return 0.0, 1.0