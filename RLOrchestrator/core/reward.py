import numpy as np
from dataclasses import dataclass


@dataclass
class RewardConfig:
    # Effectiveness: pays for log-space improvement
    w_gain: float = 40.0
    gain_ref: float = 0.02                 # reference log-improvement for scaling productivity

    # Exploration value: diversity keeps exploration alive when budget is still high
    w_diversity: float = 4.0

    # Stagnation tracking
    breakthrough_threshold: float = 0.0005 # relative improvement to reset stagnation
    stagnation_scale: float = 10.0         # smooths tanh(stagnation/scale)

    # Opportunity cost: penalize spending budget on low productivity
    w_opportunity: float = 6.0
    opportunity_power: float = 2.0         # heavier penalty when budget_remaining is high

    # Efficiency costs
    w_base_cost: float = 0.01
    w_stagnation: float = 0.5
    exploit_pressure_scale: float = 2.0    # stagnation hurts more in exploitation

    # Terminal alignment
    w_term_quality: float = 30.0
    w_term_budget: float = 12.0            # reward saving budget, scaled by quality


class ElasticRewardComputer:
    """
    Budget-aware, phase-sensitive reward shaping:
    - Effectiveness first: log-space improvement drives the main signal.
    - Exploration value: diversity is rewarded when budget is high, encouraging useful exploration.
    - Efficiency later: stagnation pressure grows with consecutive unproductive steps, stronger in exploitation.
    - Opportunity cost: low productivity while budget remains high is penalized smoothly (no hard rules).
    - Termination: quality is rewarded; saving budget is rewarded only when quality is good.
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self._prev_fitness: float = 1.0  # Normalized fitness (1.0=worst, 0.0=best)
        self._stagnation_counter: int = 0
        self._prev_phase: str = "exploration"
        self._explore_steps: int = 0
        self._exploit_steps: int = 0

    def reset(self, initial_norm_best: float):
        # Clip to ensure stability
        self._prev_fitness = np.clip(initial_norm_best, 1e-6, 1.0)
        self._stagnation_counter = 0
        self._prev_phase = "exploration"
        self._explore_steps = 0
        self._exploit_steps = 0

    def compute(
        self,
        observation: np.ndarray,
        evals_used_this_step: int,
        total_budget: int,
        total_decision_steps: int,
        terminated: bool,
    ) -> float:
        budget_remaining = float(np.clip(observation[0], 0.0, 1.0))
        active_phase = "exploitation" if observation[5] >= 0.5 else "exploration"

        if active_phase == "exploration":
            self._explore_steps += 1
        else:
            self._exploit_steps += 1

        # 1. Current Normalized Fitness (0.0 best, 1.0 worst)
        curr_fitness = float(np.clip(observation[1], 1e-6, 1.0))

        # 2. Log-Improvement (Effectiveness)
        log_improvement = np.log(self._prev_fitness) - np.log(curr_fitness)
        log_improvement = max(0.0, log_improvement)
        reward_gain = self.config.w_gain * log_improvement

        # 3. Elastic Pressure (Efficiency)
        raw_rel_improvement = (self._prev_fitness - curr_fitness) / self._prev_fitness
        if raw_rel_improvement > self.config.breakthrough_threshold:
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        phase_pressure_scale = self.config.exploit_pressure_scale if active_phase == "exploitation" else 1.0
        stagnation_factor = np.tanh(self._stagnation_counter / max(1.0, self.config.stagnation_scale))
        current_pressure = self.config.w_stagnation * stagnation_factor * phase_pressure_scale
        total_cost = self.config.w_base_cost + current_pressure

        # 4. Exploration diversity shaping (only in exploration, stronger when budget_remaining is high)
        diversity = float(np.clip(observation[4], 0.0, 1.0))
        diversity_bonus = 0.0
        if active_phase == "exploration":
            diversity_bonus = self.config.w_diversity * diversity * budget_remaining

        # 5. Opportunity cost: penalize low productivity when budget remains
        prod = min(log_improvement, self.config.gain_ref) / max(1e-8, self.config.gain_ref)
        opp_penalty = self.config.w_opportunity * (1.0 - prod) * (budget_remaining ** self.config.opportunity_power)

        # 6. Terminal Alignment
        reward_term = 0.0
        if terminated:
            quality_component = self.config.w_term_quality * (1.0 - curr_fitness)
            budget_component = self.config.w_term_budget * budget_remaining * (1.0 - curr_fitness)
            reward_term = quality_component + budget_component

        # 7. Update State
        self._prev_fitness = curr_fitness
        self._prev_phase = active_phase

        reward = (
            reward_gain
            + diversity_bonus
            + reward_term
            - total_cost
            - opp_penalty
        )

        return float(reward)
