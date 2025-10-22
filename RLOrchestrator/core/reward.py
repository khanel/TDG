"""
Generalized reward computation for RL environment.
Problem-agnostic, based on solver metrics and phase transitions.
"""

from typing import Optional
import numpy as np


class RewardComputer:
    """Phase-aware reward shaping for exploration → exploitation search."""

    def __init__(self, problem_bounds: dict, *, clip_range: tuple[float, float] = (-1.0, 1.0)):
        self.lower_bound = float(problem_bounds.get("lower_bound", 0.0))
        self.upper_bound = float(problem_bounds.get("upper_bound", 1.0))
        lo, hi = clip_range
        if lo > hi:
            lo, hi = hi, lo
        self._clip_min = float(lo)
        self._clip_max = float(hi)
        self._fitness_range = max(1e-9, self.upper_bound - self.lower_bound)
        # Weight configuration balances exploration diversity, exploitation gains, and efficiency.
        self.w = {
            "exploration_improvement": 0.25,
            "exploration_success": 0.20,
            "exploration_entropy": 0.20,
            "exploration_stagnation": 0.30,
            "exploitation_progress": 0.40,
            "exploitation_quality": 0.30,
            "exploitation_stagnation": 0.25,
            "switch_to_exploitation": 0.40,
            "early_switch_penalty": 0.30,
            "late_switch_penalty": 0.20,
            "terminate_quality": 0.40,
            "terminate_timing": 0.30,
            "early_termination_penalty": 0.30,
            "bad_solution_penalty": 0.30,
            "budget_overuse_penalty": 0.30,
            "idle_penalty": 0.20,
        }

    def compute(
        self,
        *,
        action: int,
        phase: str,
        improvement: float,
        terminated: bool,
        observation: np.ndarray,
    ) -> float:
        """
        Compute a shaped reward after the solver has advanced.

        Args:
            action: Agent action taken before this transition (0=continue, 1=advance phase).
            phase: Phase prior to taking the action ("exploration" or "exploitation").
            improvement: Fitness delta (prev_best - curr_best).
            terminated: Whether the episode ended after this transition.
            observation: Post-transition observation vector (length 7 as defined in ObservationComputer).
        """
        obs = np.asarray(observation, dtype=float)
        normalized_best = float(np.clip(obs[1], 0.0, 1.0))
        frontier_improved = float(np.clip(obs[2], 0.0, 1.0))
        success_rate = float(np.clip(obs[3], 0.0, 1.0))
        elite_entropy = float(np.clip(obs[4], 0.0, 1.0))
        stagnation = float(np.clip(obs[5], 0.0, 1.0))
        budget_ratio = float(np.clip(obs[6], 0.0, 1.0))

        normalized_improvement = float(np.clip(improvement / self._fitness_range, -1.0, 1.0))
        solution_quality = 1.0 - normalized_best  # higher is better

        if action == 1:
            return self._compute_switch_reward(
                phase=phase,
                normalized_improvement=normalized_improvement,
                solution_quality=solution_quality,
                success_rate=success_rate,
                stagnation=stagnation,
                frontier_improved=frontier_improved,
                budget_ratio=budget_ratio,
                terminated=terminated,
            )

        reward = 0.0
        # --- Phase-specific shaping ---
        if phase == "exploration":
            reward += self.w["exploration_improvement"] * frontier_improved
            reward += self.w["exploration_success"] * (success_rate - 0.5)
            reward += self.w["exploration_entropy"] * (elite_entropy - 0.5)
            reward -= self.w["exploration_stagnation"] * stagnation
            if action == 0 and normalized_improvement < 0.0:
                reward += normalized_improvement * 0.1  # discourage regressions while continuing exploration
        else:  # exploitation
            reward += self.w["exploitation_progress"] * normalized_improvement
            reward += self.w["exploitation_quality"] * (solution_quality - 0.5)
            reward -= self.w["exploitation_stagnation"] * stagnation

        # --- Efficiency penalties ---
        if action == 0:
            if normalized_improvement <= 0.0 and stagnation > 0.7:
                reward -= self.w["idle_penalty"] * stagnation
            if budget_ratio > 0.98:
                reward -= self.w["budget_overuse_penalty"] * (budget_ratio - 0.98)
        else:
            if budget_ratio > 0.99 and not terminated:
                reward -= self.w["budget_overuse_penalty"] * (budget_ratio - 0.99)

        return float(np.clip(reward, self._clip_min, self._clip_max))

    def _compute_switch_reward(
        self,
        *,
        phase: str,
        normalized_improvement: float,
        solution_quality: float,
        success_rate: float,
        stagnation: float,
        frontier_improved: float,
        budget_ratio: float,
        terminated: bool,
    ) -> float:
        """
        Dedicated reward logic for action == 1 (phase change / termination).

        Goals:
        - Reward well-timed switches/terminations.
        - Strongly penalize premature decisions.
        - Strongly penalize wasting the decision budget.
        - Do not clip severe penalties so the agent learns the cost.
        """
        # Encourage switches only after exploration has stalled and success has dropped.
        if phase == "exploration":
            readiness = 0.7 * stagnation + 0.3 * max(0.0, 0.6 - success_rate)
            reward = 0.0

            # Positive credit when readiness comfortably exceeds the threshold.
            readiness_threshold = 0.45
            readiness_gap = readiness - readiness_threshold
            if readiness_gap >= 0.0:
                reward += 4.0 * readiness_gap
                reward += 1.5 * max(0.0, solution_quality - 0.5)
                reward += 1.0 * max(0.0, frontier_improved - 0.5)
            else:
                # Hard penalty for premature switching.
                reward -= 8.0 * (-readiness_gap)

            # Penalize budget waste from switching too late.
            if budget_ratio > 0.9:
                reward -= 6.0 * (budget_ratio - 0.9 + 0.02)

            # Discourage switching when recent improvement is negative.
            if normalized_improvement < 0.0:
                reward += 0.5 * normalized_improvement

            return reward

        # Termination decision during exploitation.
        reward = 0.0
        quality_threshold = 0.6
        budget_min_threshold = 0.55

        if solution_quality >= quality_threshold:
            reward += 5.0 * (solution_quality - quality_threshold)
            reward += 2.0 * max(0.0, normalized_improvement)
        else:
            reward -= 9.0 * (quality_threshold - solution_quality)

        if budget_ratio < budget_min_threshold:
            # Severe penalty for early termination.
            reward -= 7.0 * (budget_min_threshold - budget_ratio)
        else:
            reward += 2.5 * (budget_ratio - budget_min_threshold)

        if budget_ratio > 0.97:
            # Hard penalty for running the clock down.
            reward -= 6.5 * (budget_ratio - 0.97 + 0.03)

        if normalized_improvement < 0.0:
            reward -= 2.5 * abs(normalized_improvement)

        if not terminated:
            # Termination action should have ended the episode.
            reward -= 3.0

        return reward
