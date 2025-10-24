"""
Generalized reward computation for RL environment.
Problem-agnostic, based on solver metrics and phase transitions.
"""

from typing import Optional
import numpy as np


class RewardComputer:
    """Phase-aware reward shaping for exploration → exploitation search."""

    def __init__(self, problem_bounds: dict, *, clip_range: tuple[float, float] = (-1.0, 1.0)):
        self.lower_bound, self.upper_bound = self._extract_bounds(problem_bounds)
        lo, hi = clip_range
        if lo > hi:
            lo, hi = hi, lo
        self._clip_min = float(lo)
        self._clip_max = float(hi)
        self._fitness_range = max(1e-9, self.upper_bound - self.lower_bound)

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
        Compute reward after the solver has advanced.

        Args:
            action: Agent action taken before this transition.
            phase: Phase prior to taking the action.
            improvement: Best-fitness delta observed during the step (prev - curr).
            terminated: Whether the environment terminated after the action.
            observation: Observation after stepping.
            prev_observation: Observation before the action.
            steps_run: Number of solver steps executed this decision.
            switched: Whether the action triggered exploration → exploitation.
            phase_after: Phase after the transition (if still running).
        """
        obs = np.asarray(observation, dtype=float)
        prev_obs = np.asarray(prev_observation, dtype=float) if prev_observation is not None else obs
        normalized_improvement = float(np.clip(improvement / self._fitness_range, -1.0, 1.0))

        if action == 0:
            return self._score_continue(prev_obs, obs, normalized_improvement, steps_run, phase)

        if phase == "exploration" and switched:
            return self._score_switch(prev_obs, normalized_improvement)
        # phase == "exploitation" termination (or unexpected switch failure)
        return self._score_terminate(prev_obs, obs, normalized_improvement, terminated, phase_after)

    def _score_continue(
        self,
        prev_obs: np.ndarray,
        curr_obs: np.ndarray,
        normalized_improvement: float,
        steps_run: int,
        phase: str,
    ) -> float:
        budget = float(np.clip(curr_obs[6], 0.0, 1.0))
        stagnation = float(np.clip(curr_obs[5], 0.0, 1.0))
        entropy = float(np.clip(curr_obs[4], 0.0, 1.0))

        reward = 0.0

        if phase == "exploration":
            reward += 0.7 * max(normalized_improvement, 0.0)
            reward -= 0.4 * max(-normalized_improvement, 0.0)
            reward += 0.3 * (entropy - 0.5)
            reward -= 0.4 * max(0.0, stagnation - 0.55)
            reward -= 0.35 * max(0.0, budget - 0.8)
        else:  # exploitation
            reward += 0.8 * max(normalized_improvement, 0.0)
            reward -= 0.6 * max(-normalized_improvement, 0.0)
            reward -= 0.5 * max(0.0, stagnation - 0.45)
            reward -= 0.4 * max(0.0, budget - 0.95)

        if steps_run == 0:
            reward -= 0.2

        return float(np.clip(reward, self._clip_min, self._clip_max))

    def _score_switch(
        self,
        prev_obs: np.ndarray,
        normalized_improvement: float,
    ) -> float:
        budget = float(np.clip(prev_obs[6], 0.0, 1.0))
        stagnation = float(np.clip(prev_obs[5], 0.0, 1.0))
        success = float(np.clip(prev_obs[3], 0.0, 1.0))
        diversity = float(np.clip(prev_obs[4], 0.0, 1.0))

        readiness = 0.5 * max(0.0, budget - 0.55) + 0.5 * max(0.0, stagnation - 0.5)
        readiness += 0.3 * max(0.0, 0.35 - success)

        penalty = 0.6 * max(0.0, 0.5 - budget)
        penalty += 0.5 * max(0.0, 0.4 - stagnation)
        penalty += 0.3 * max(0.0, diversity - 0.65)
        penalty += 0.4 * max(normalized_improvement, 0.0)

        reward = readiness - penalty
        reward += 0.3 * max(0.0, -normalized_improvement)

        return float(np.clip(reward, self._clip_min, self._clip_max))

    def _score_terminate(
        self,
        prev_obs: np.ndarray,
        curr_obs: np.ndarray,
        normalized_improvement: float,
        terminated: bool,
        phase_after: Optional[str],
    ) -> float:
        stagnation_prev = float(np.clip(prev_obs[5], 0.0, 1.0))
        budget_prev = float(np.clip(prev_obs[6], 0.0, 1.0))
        readiness = 0.6 * max(0.0, budget_prev - 0.75) + 0.6 * max(0.0, stagnation_prev - 0.5)
        readiness += 0.4 * max(0.0, -normalized_improvement)

        penalty = 0.6 * max(0.0, 0.65 - budget_prev)
        penalty += 0.5 * max(0.0, 0.4 - stagnation_prev)
        penalty += 0.5 * max(0.0, normalized_improvement)

        reward = readiness - penalty
        if not terminated:
            reward -= 0.6
        if phase_after and phase_after != "exploitation":
            reward -= 0.2

        return float(np.clip(reward, self._clip_min, self._clip_max))

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
