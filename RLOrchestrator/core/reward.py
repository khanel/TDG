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
        efficiency_penalty: float = 0.001,
        logger: logging.Logger,
    ):
        """
        Initializes the reward computer.
        Args:
            problem_bounds: Dictionary with fitness bounds ('lower_bound', 'upper_bound').
            clip_range: Tuple to clip the final reward.
            efficiency_penalty: Small constant penalty per step to encourage efficiency.
        """
        self.logger = logger
        self.lower_bound, self.upper_bound = self._extract_bounds(problem_bounds)
        lo, hi = clip_range
        if lo > hi:
            lo, hi = hi, lo
        self._clip_min = float(lo)
        self._clip_max = float(hi)
        self._fitness_range = max(1e-9, self.upper_bound - self.lower_bound)
        self.efficiency_penalty = float(efficiency_penalty)

        # Keep initialization logs at debug level to avoid noisy files
        self.logger.debug("RewardComputer initialized")
        self.logger.debug(f"  problem_bounds: {problem_bounds}")
        self.logger.debug(f"  clip_range: {clip_range}")
        self.logger.debug(f"  efficiency_penalty: {self.efficiency_penalty}")

        # --- Reward shaping hyperparameters (tunable, but with sane defaults) ---
        # Budget schedules for quality vs exploration
        self._quality_exp: float = 1.0  # a in w_q(B) = (1-B)^a
        self._explore_exp: float = 1.0  # b in w_e(B) = B^b

        # Switch advantage parameters
        self._switch_stagnation_threshold: float = 0.65  # S_sw
        self._switch_alpha: float = 0.4                 # gain on readiness
        self._switch_beta: float = 0.6                  # cost multiplier
        self._switch_early_power: float = 2.0           # exponent on B in early cost

        # Termination advantage parameters
        self._term_stagnation_threshold: float = 0.85   # S_term
        self._term_eta: float = 0.8                     # gain on benefit
        self._term_zeta: float = 0.8                    # penalty on unused budget
        self._term_cost_power: float = 2.0              # exponent on B in termination cost
        self._term_late_power: float = 1.0              # exponent on (1-B) in term benefit

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

        # --- Helpers: budget and deltas from observations ---
        # Use previous observation for budget weighting of the decision being made now.
        B_prev = float(np.clip(prev_obs[0], 0.0, 1.0)) if prev_obs is not None and len(prev_obs) >= 1 else float(np.clip(obs[0], 0.0, 1.0))
        B_curr = float(np.clip(obs[0], 0.0, 1.0))
        F_prev = float(np.clip(prev_obs[1], 0.0, 1.0)) if prev_obs is not None and len(prev_obs) >= 2 else float(np.clip(obs[1], 0.0, 1.0))
        F_curr = float(np.clip(obs[1], 0.0, 1.0))
        V_prev = float(np.clip(prev_obs[2], -1.0, 1.0)) if prev_obs is not None and len(prev_obs) >= 3 else float(np.clip(obs[2], -1.0, 1.0))
        V_curr = float(np.clip(obs[2], -1.0, 1.0))
        S_prev = float(np.clip(prev_obs[3], 0.0, 1.0)) if prev_obs is not None and len(prev_obs) >= 4 else float(np.clip(obs[3], 0.0, 1.0))
        S_curr = float(np.clip(obs[3], 0.0, 1.0))
        C_prev = float(np.clip(prev_obs[4], 0.0, 1.0)) if prev_obs is not None and len(prev_obs) >= 5 else float(np.clip(obs[4], 0.0, 1.0))
        C_curr = float(np.clip(obs[4], 0.0, 1.0))

        dC = C_prev - C_curr
        dV = V_curr - V_prev

        # --- 1. Progress Reward (R_progress) ---
        # Normalized fitness delta. Positive for improvement.
        normalized_improvement = float(np.clip(improvement / self._fitness_range, -1.0, 1.0))
        w_quality = (1.0 - B_prev) ** self._quality_exp
        progress_reward = w_quality * normalized_improvement

        # --- 2. Phase-aware Diversity Shaping (R_div) ---
        # Encourage dispersion in exploration; encourage consolidation in exploitation.
        w_explore = (B_prev ** self._explore_exp)
        if str(phase) == "exploration":
            diversity_reward = w_explore * dC  # lower concentration (higher diversity) is good
        else:
            diversity_reward = w_explore * (-dC)  # higher concentration (more focus) is good

        # --- 3. Switch Advantage (R_switch) ---
        switch_reward = 0.0
        if int(action) == 1 and bool(switched) and str(phase) == "exploration":
            # Readiness increases with stagnation above threshold, decreasing velocity, and low concentration
            stagnation_ready = max(0.0, S_prev - self._switch_stagnation_threshold)
            velocity_cooling = max(0.0, -dV)
            velocity_cooling = float(np.clip(velocity_cooling, 0.0, 1.0))  # cap magnitude
            diversity_ready = max(0.0, 1.0 - C_prev)
            ready_switch = stagnation_ready * velocity_cooling * diversity_ready

            # Early switching cost scales with remaining budget and low stagnation
            early_cost = (B_prev ** self._switch_early_power) * max(0.0, self._switch_stagnation_threshold - S_prev)

            switch_reward = (self._switch_alpha * ready_switch) - (self._switch_beta * early_cost)

        # --- 4. Termination Advantage (R_term) ---
        term_reward = 0.0
        if int(action) == 1 and bool(terminated) and str(phase) == "exploitation":
            # Benefit prefers good solutions, high stagnation, and late budget
            benefit = (1.0 - F_curr) * max(0.0, S_curr - self._term_stagnation_threshold) * ((1.0 - B_curr) ** self._term_late_power)
            # Cost penalizes unused budget
            cost_unused = (B_curr ** self._term_cost_power)
            term_reward = (self._term_eta * benefit) - (self._term_zeta * cost_unused)

        # --- 5. Total Reward Combination ---
        # R_total = w_q(B)*R_progress + w_e(B)*R_diversity - C + switch_adv + term_adv
        total_reward = (
            progress_reward
            + diversity_reward
            - self.efficiency_penalty
            + switch_reward
            + term_reward
        )

        final_reward = float(np.clip(total_reward, self._clip_min, self._clip_max))

        self.logger.debug("Reward calculation:")
        self.logger.debug(f"  - improvement_raw: {improvement:.4f}, normalized: {normalized_improvement:.4f}")
        self.logger.debug(f"  - budgets: B_prev={B_prev:.4f}, B_curr={B_curr:.4f}")
        self.logger.debug(f"  - progress_reward (w_q * dF): {progress_reward:.4f} (w_q={w_quality:.4f})")
        self.logger.debug(f"  - diversity: dC={dC:.4f}, phase={phase}, R_div={diversity_reward:.4f} (w_e={w_explore:.4f})")
        if int(action) == 1 and bool(switched) and str(phase) == "exploration":
            self.logger.debug(f"  - switch_adv: {switch_reward:.4f} (S_prev={S_prev:.3f}, dV={dV:.3f}, C_prev={C_prev:.3f})")
        if int(action) == 1 and bool(terminated) and str(phase) == "exploitation":
            self.logger.debug(f"  - term_adv: {term_reward:.4f} (F_curr={F_curr:.3f}, S_curr={S_curr:.3f})")
        self.logger.debug(f"  - efficiency_penalty: {self.efficiency_penalty:.4f}")
        self.logger.debug(f"  - total_reward (before clip): {total_reward:.4f}")
        self.logger.debug(f"  - final_reward (after clip): {final_reward:.4f}")

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
