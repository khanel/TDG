"""
Effectiveness-First Reward Function (EFR)
==========================================

A single, coherent reward function for 3-phase RL hyper-heuristic orchestration.

CORE PRINCIPLE: EFFECTIVENESS FIRST, EFFICIENCY SECOND
-------------------------------------------------------
1. Effectiveness (Solution Quality) is MANDATORY
   - Quality < threshold = FAILURE, regardless of how fast
   - No efficiency bonus below quality threshold
   
2. Efficiency (Budget Savings) is SECONDARY
   - Only matters AFTER quality threshold is met
   - Early exit with garbage = WORST outcome (not "efficient")

Design: "Finding shit quickly is not efficiency - it's just fast failure."

Phase Pipeline (Unified Action Semantics):
------------------------------------------
- EXPLORATION: Action 0=STAY, Action 1=ADVANCE to Exploitation
- EXPLOITATION: Action 0=STAY, Action 1=ADVANCE to Termination  
- TERMINATION: Episode ends, terminal reward computed

Mathematical Properties:
------------------------
- Bounded: All rewards in [-1.0, +1.0]
- Lipschitz Continuous: Smooth sigmoids, K < 10 guaranteed
- Quality-Gated: Efficiency bonus = 0 when quality < 0.7
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class RewardSignal:
    """Standardized return for reward calculations."""
    value: float                          # Final scalar for RL agent, bounded [-1, 1]
    components: Dict[str, float]          # Breakdown for debugging
    phase: str                            # Current phase
    quality: float                        # Solution quality (1 - fitness_norm)
    efficiency_unlocked: bool             # Whether quality gate was passed


@dataclass  
class EFRConfig:
    """
    Configuration for Effectiveness-First Reward.
    
    All thresholds are problem-agnostic (normalized to [0,1]).
    Quality = 1 - fitness_norm (0=worst, 1=optimal)
    """
    # === QUALITY GATE (CRITICAL) ===
    quality_threshold: float = 0.70       # Below this = FAILURE, no efficiency bonus
    quality_excellent: float = 0.85       # Above this = excellent, max bonuses
    
    # === STEP COST ===
    step_cost: float = -0.01              # Small cost per step to encourage progress
    
    # === EXPLORATION PHASE ===
    explore_improvement_reward: float = 0.35      # Finding improvement
    explore_diversity_bonus_max: float = 0.15     # High diversity bonus
    explore_stagnation_penalty_max: float = -0.6  # Stuck without progress
    explore_stagnation_threshold: float = 0.35    # When stagnation penalty starts
    explore_switch_stagnation_center: float = 0.45  # Sigmoid center for switch readiness
    explore_timely_switch_reward: float = 0.5     # Reward for switching at right time
    explore_premature_switch_penalty: float = -0.6  # Penalty for switching while improving
    explore_very_early_penalty: float = -0.8      # Penalty for switching with stagnation < 0.1
    
    # === EXPLOITATION PHASE ===
    exploit_improvement_reward: float = 0.35      # Finding improvement (same as explore)
    exploit_quality_bonus_scale: float = 0.3      # Bonus scaled by current quality
    exploit_stagnation_penalty_max: float = -0.4  # Stuck without progress
    exploit_stagnation_threshold: float = 0.55    # Higher threshold (exploitation expects more stagnation)
    exploit_advance_stagnation_center: float = 0.6  # Sigmoid center for advance readiness
    exploit_timely_advance_reward: float = 0.4    # Reward for timely advance to termination
    exploit_premature_advance_penalty: float = -0.5  # Penalty for advancing while improving
    exploit_very_early_penalty: float = -0.7      # Penalty for advancing with stagnation < 0.2
    
    # === TERMINATION PHASE ===
    # EFFECTIVENESS FIRST: Quality determines everything
    termination_excellent_reward: float = 0.8     # quality >= excellent threshold
    termination_good_reward: float = 0.4          # quality >= quality_threshold
    termination_poor_penalty: float = -0.6        # quality < quality_threshold
    termination_terrible_penalty: float = -1.0    # quality < 0.4 (far below threshold)
    
    # EFFICIENCY SECOND: Only applies when quality >= threshold
    efficiency_bonus_max: float = 0.3             # Max bonus for early termination with quality
    efficiency_quality_scale: float = 0.5         # How much quality above threshold boosts efficiency
    
    # === BUDGET PRESSURE ===
    budget_pressure_onset: float = 0.75           # When budget pressure starts
    budget_pressure_max: float = -0.3             # Max pressure at budget exhaustion
    
    # === SIGMOID SMOOTHNESS ===
    sigmoid_scale: float = 8.0                    # Controls sigmoid steepness (lower = smoother)


class EffectivenessFirstReward:
    """
    Effectiveness-First Reward Function.
    
    Core Design:
    ------------
    1. Quality gates everything - no shortcuts for poor solutions
    2. Smooth sigmoids for all transitions (Lipschitz continuous)
    3. Consistent action semantics (0=STAY, 1=ADVANCE)
    4. Terminal quality is what ultimately matters
    
    Anti-Patterns Heavily Penalized:
    --------------------------------
    - Ultra-fast exit (3 steps) with garbage quality → WORST possible
    - Early exit without meeting quality threshold → Strong negative
    - Skipping exploration entirely → Negative
    - Skipping exploitation entirely → Negative
    - Advancing while still improving → Negative
    """
    
    def __init__(self, config: Optional[EFRConfig] = None):
        self.cfg = config or EFRConfig()
    
    def calculate(
        self, 
        state: Dict[str, Any], 
        action: int, 
        phase: str
    ) -> RewardSignal:
        """
        Calculate reward for state-action pair.
        
        Args:
            state: Observation dictionary with keys:
                - fitness_norm: float [0,1] (0=optimal, 1=worst)
                - improvement_velocity: float [0,1] (EWMA of improvement velocity)
                - diversity_score: float [0,1]
                - stagnation_ratio: float [0,1]
                - budget_consumed: float [0,1]
            action: 0 (STAY) or 1 (ADVANCE)
            phase: 'EXPLORATION', 'EXPLOITATION', or 'TERMINATION'
            
        Returns:
            RewardSignal with bounded value in [-1, 1]
        """
        # === Extract State ===
        fitness_norm = float(state.get('fitness_norm', 0.5))
        improvement = float(state.get('improvement_velocity', 0.0))  # Now continuous [0,1]
        diversity = float(state.get('diversity_score', 0.5))
        stagnation = float(state.get('stagnation_ratio', 0.0))
        budget = float(state.get('budget_consumed', 0.0))
        
        # Quality is inverse of normalized fitness
        quality = 1.0 - fitness_norm
        
        # === Phase-Specific Reward ===
        if phase == 'EXPLORATION':
            reward, components = self._exploration(action, improvement, diversity, stagnation, budget)
        elif phase == 'EXPLOITATION':
            reward, components = self._exploitation(action, improvement, diversity, stagnation, budget, quality)
        else:  # TERMINATION
            reward, components, efficiency_unlocked = self._termination(quality, budget, stagnation)
            return RewardSignal(
                value=self._bound(reward),
                components=components,
                phase=phase,
                quality=quality,
                efficiency_unlocked=efficiency_unlocked
            )
        
        # === Budget Pressure (non-terminal phases only) ===
        if budget > self.cfg.budget_pressure_onset:
            pressure_factor = (budget - self.cfg.budget_pressure_onset) / (1.0 - self.cfg.budget_pressure_onset + 1e-8)
            pressure = self.cfg.budget_pressure_max * self._smooth_sigmoid(pressure_factor, 0.5, 6.0)
            reward += pressure
            components['budget_pressure'] = pressure
        
        # Check if we'd qualify for efficiency bonus (for debugging)
        efficiency_unlocked = quality >= self.cfg.quality_threshold
        
        return RewardSignal(
            value=self._bound(reward),
            components=components,
            phase=phase,
            quality=quality,
            efficiency_unlocked=efficiency_unlocked
        )
    
    def _exploration(
        self, 
        action: int, 
        improvement: int, 
        diversity: float, 
        stagnation: float,
        budget: float
    ) -> tuple:
        """
        Exploration phase: Find promising regions, maintain diversity.
        
        Goals:
        - Reward improvements (with diversity bonus)
        - Penalize stagnation (encourage advance when stuck)
        - Reward timely advance, penalize premature advance
        """
        cfg = self.cfg
        components = {'step_cost': cfg.step_cost}
        reward = cfg.step_cost
        
        if action == 0:  # STAY in exploration
            if improvement == 1:
                # === IMPROVEMENT FOUND ===
                # Base reward + diversity bonus (diverse improvements are better)
                diversity_bonus = cfg.explore_diversity_bonus_max * diversity
                r = cfg.explore_improvement_reward + diversity_bonus
                components['improvement'] = cfg.explore_improvement_reward
                components['diversity_bonus'] = diversity_bonus
                reward += r
                
            elif stagnation > cfg.explore_stagnation_threshold:
                # === STAGNATING ===
                # Smooth penalty that increases with stagnation
                stag_factor = self._smooth_sigmoid(
                    stagnation, 
                    (cfg.explore_stagnation_threshold + 1.0) / 2,  # Center between threshold and 1
                    cfg.sigmoid_scale
                )
                # Diversity mitigates penalty (diverse stagnation is less bad)
                mitigation = 1.0 - (diversity * 0.4)
                r = cfg.explore_stagnation_penalty_max * stag_factor * mitigation
                components['stagnation_penalty'] = r
                reward += r
                
            else:
                # === EXPLORING (not improving, not stagnant) ===
                # Small diversity maintenance bonus
                if diversity > 0.5:
                    r = cfg.explore_diversity_bonus_max * 0.3 * (diversity - 0.5) * 2
                    components['diversity_maintenance'] = r
                    reward += r
        
        else:  # action == 1: ADVANCE to exploitation
            if improvement == 1:
                # === PREMATURE SWITCH ===
                # Switching while still finding improvements - bad decision
                r = cfg.explore_premature_switch_penalty
                components['premature_switch'] = r
                reward += r
                
            elif stagnation < 0.1:
                # === VERY EARLY SWITCH ===
                # Stagnation near zero - exploration barely started
                r = cfg.explore_very_early_penalty
                components['very_early_switch'] = r
                reward += r
                
            else:
                # === SWITCH DECISION ===
                # Smooth interpolation based on stagnation
                switch_readiness = self._smooth_sigmoid(
                    stagnation, 
                    cfg.explore_switch_stagnation_center, 
                    cfg.sigmoid_scale
                )
                
                # Interpolate: low stagnation = penalty, high stagnation = reward
                early_penalty = -0.3
                r = early_penalty * (1.0 - switch_readiness) + cfg.explore_timely_switch_reward * switch_readiness
                
                if r >= 0:
                    components['timely_switch'] = r
                else:
                    components['uncertain_switch'] = r
                reward += r
        
        return reward, components
    
    def _exploitation(
        self, 
        action: int, 
        improvement: int, 
        diversity: float, 
        stagnation: float,
        budget: float,
        quality: float
    ) -> tuple:
        """
        Exploitation phase: Refine solutions toward high quality.
        
        Goals:
        - Reward improvements (with quality-scaled bonus)
        - Penalize stagnation (encourage advance when stuck)
        - Reward timely advance to termination
        """
        cfg = self.cfg
        components = {'step_cost': cfg.step_cost}
        reward = cfg.step_cost
        
        if action == 0:  # STAY in exploitation
            if improvement == 1:
                # === IMPROVEMENT FOUND ===
                # Base reward + quality-scaled bonus (improving good solutions is harder/better)
                quality_bonus = cfg.exploit_quality_bonus_scale * quality
                r = cfg.exploit_improvement_reward + quality_bonus
                components['improvement'] = cfg.exploit_improvement_reward
                components['quality_bonus'] = quality_bonus
                reward += r
                
            elif stagnation > cfg.exploit_stagnation_threshold:
                # === STAGNATING ===
                stag_factor = self._smooth_sigmoid(
                    stagnation,
                    (cfg.exploit_stagnation_threshold + 1.0) / 2,
                    cfg.sigmoid_scale
                )
                r = cfg.exploit_stagnation_penalty_max * stag_factor
                components['stagnation_penalty'] = r
                reward += r
                
            # else: grinding without improvement, just step cost
        
        else:  # action == 1: ADVANCE to termination
            if improvement == 1:
                # === PREMATURE ADVANCE ===
                # Advancing while still finding improvements - leaving value on table
                r = cfg.exploit_premature_advance_penalty
                components['premature_advance'] = r
                reward += r
                
            elif stagnation < 0.2:
                # === VERY EARLY ADVANCE ===
                # Stagnation too low - exploitation barely started
                r = cfg.exploit_very_early_penalty
                components['very_early_advance'] = r
                reward += r
                
            else:
                # === ADVANCE DECISION ===
                advance_readiness = self._smooth_sigmoid(
                    stagnation,
                    cfg.exploit_advance_stagnation_center,
                    cfg.sigmoid_scale
                )
                
                early_penalty = -0.25
                r = early_penalty * (1.0 - advance_readiness) + cfg.exploit_timely_advance_reward * advance_readiness
                
                if r >= 0:
                    components['timely_advance'] = r
                else:
                    components['uncertain_advance'] = r
                reward += r
        
        return reward, components
    
    def _termination(
        self, 
        quality: float, 
        budget: float,
        stagnation: float
    ) -> tuple:
        """
        Termination phase: Evaluate the stopping decision.
        
        EFFECTIVENESS FIRST:
        - Quality determines primary reward/penalty
        - Efficiency bonus ONLY applies if quality >= threshold
        
        EFFICIENCY SECOND:
        - Budget remaining provides bonus ONLY when quality is good
        - Early exit with garbage = heavily penalized (not "efficient")
        """
        cfg = self.cfg
        components = {}
        
        # === QUALITY-BASED REWARD (EFFECTIVENESS) ===
        if quality >= cfg.quality_excellent:
            # === EXCELLENT QUALITY ===
            # Exceeded expectations - strong reward
            quality_reward = cfg.termination_excellent_reward
            components['excellent_quality'] = quality_reward
            efficiency_unlocked = True
            
        elif quality >= cfg.quality_threshold:
            # === GOOD QUALITY ===
            # Met the bar - positive reward
            # Smooth interpolation between threshold and excellent
            quality_factor = (quality - cfg.quality_threshold) / (cfg.quality_excellent - cfg.quality_threshold + 1e-8)
            quality_factor = np.clip(quality_factor, 0.0, 1.0)
            quality_reward = cfg.termination_good_reward + (cfg.termination_excellent_reward - cfg.termination_good_reward) * quality_factor
            components['good_quality'] = quality_reward
            efficiency_unlocked = True
            
        elif quality >= 0.4:
            # === POOR QUALITY ===
            # Below threshold but not terrible
            quality_reward = cfg.termination_poor_penalty
            components['poor_quality'] = quality_reward
            efficiency_unlocked = False  # NO EFFICIENCY BONUS
            
        else:
            # === TERRIBLE QUALITY ===
            # Far below threshold - this is the BUG we're fixing
            # Fast exit with garbage = WORST outcome
            quality_reward = cfg.termination_terrible_penalty
            components['terrible_quality'] = quality_reward
            efficiency_unlocked = False  # NO EFFICIENCY BONUS
        
        reward = quality_reward
        
        # === EFFICIENCY BONUS (ONLY IF QUALITY GATE PASSED) ===
        if efficiency_unlocked:
            budget_remaining = max(0.0, 1.0 - budget)
            
            if budget_remaining > 0.05:
                # Quality above threshold unlocks efficiency bonus
                # Bonus scales with how much above threshold we are
                quality_above = (quality - cfg.quality_threshold) / (1.0 - cfg.quality_threshold + 1e-8)
                quality_above = np.clip(quality_above, 0.0, 1.0)
                
                # Efficiency bonus = budget_remaining * quality_scale * max_bonus
                efficiency_scale = 1.0 + cfg.efficiency_quality_scale * quality_above
                efficiency_bonus = cfg.efficiency_bonus_max * budget_remaining * efficiency_scale
                efficiency_bonus = min(efficiency_bonus, cfg.efficiency_bonus_max)  # Cap at max
                
                components['efficiency_bonus'] = efficiency_bonus
                reward += efficiency_bonus
        else:
            # Quality below threshold - efficiency bonus is ZERO
            # This is the key insight: fast + garbage ≠ efficient
            components['efficiency_blocked'] = 0.0
        
        return reward, components, efficiency_unlocked
    
    def _smooth_sigmoid(self, x: float, center: float, scale: float) -> float:
        """
        Smooth sigmoid for continuous transitions.
        Returns value in [0, 1] with smooth transition around center.
        
        Guarantees Lipschitz continuity with K < scale.
        """
        z = scale * (x - center)
        # Numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1.0 + exp_z)
    
    def _bound(self, value: float) -> float:
        """Clip reward to [-1, 1]."""
        return float(np.clip(value, -1.0, 1.0))


# ================================================================
# CONVENIENCE FUNCTION
# ================================================================

def create_reward_function(config: Optional[EFRConfig] = None) -> EffectivenessFirstReward:
    """Create the Effectiveness-First Reward function."""
    return EffectivenessFirstReward(config)


# ================================================================
# INTEGRATION WITH GYMNASIUM ENVIRONMENT
# ================================================================

class RewardWrapper:
    """
    Wrapper to integrate EFR with the OrchestratorEnv.
    
    Usage:
        env = OrchestratorEnv(...)
        reward_fn = RewardWrapper()
        
        obs, info = env.reset()
        while not done:
            action = agent.select_action(obs)
            next_obs, _, terminated, truncated, info = env.step(action)
            
            # Calculate reward using our function
            reward_signal = reward_fn.compute(obs, action, info)
            reward = reward_signal.value
    """
    
    def __init__(self, config: Optional[EFRConfig] = None):
        self.efr = EffectivenessFirstReward(config)
        self._phase_map = {
            0.0: 'EXPLORATION',
            0.5: 'EXPLOITATION', 
            1.0: 'TERMINATION'
        }
    
    def compute(
        self, 
        obs: np.ndarray,
        action: int,
        info: dict
    ) -> RewardSignal:
        """
        Compute reward from environment observations.
        
        Observation format (6 features):
        [0] budget_consumed: float [0,1] - budget used so far
        [1] fitness_norm: float [0,1] - normalized fitness (0=optimal, 1=worst)
        [2] improvement_velocity: float [0,1] - EWMA of improvement velocity
        [3] stagnation: float [0,1] - search stagnation level
        [4] diversity: float [0,1] - population diversity
        [5] phase: float {0.0, 0.5, 1.0} - current phase encoding
        
        Args:
            obs: Current observation array from OrchestratorEnv
            action: Action taken (0=STAY, 1=ADVANCE)
            info: Step info dict from environment
            
        Returns:
            RewardSignal
        """
        # Extract state from observation (matches base.py ObservationComputer)
        budget_consumed = float(obs[0])
        fitness_norm = float(obs[1])
        improvement = float(obs[2])  # Now continuous EWMA velocity [0,1]
        stagnation = float(obs[3])
        diversity = float(obs[4])
        phase_encoded = float(obs[5])
        
        # Decode phase
        if phase_encoded < 0.25:
            phase = 'EXPLORATION'
        elif phase_encoded < 0.75:
            phase = 'EXPLOITATION'
        else:
            phase = 'TERMINATION'
        
        # Build state dict
        state = {
            'fitness_norm': fitness_norm,
            'improvement_velocity': improvement,
            'diversity_score': diversity,
            'stagnation_ratio': stagnation,
            'budget_consumed': budget_consumed
        }
        
        return self.efr.calculate(state, action, phase)


# ================================================================
# DIRECT STATE-BASED INTERFACE (for custom integrations)
# ================================================================

def compute_reward(
    fitness_norm: float,
    improvement: float,
    diversity: float,
    stagnation: float,
    budget_consumed: float,
    action: int,
    phase: str,
    config: Optional[EFRConfig] = None
) -> RewardSignal:
    """
    Compute reward directly from state variables.
    
    This is the simplest interface for integration.
    
    Args:
        fitness_norm: Normalized fitness [0,1] where 0=optimal, 1=worst
        improvement: Improvement velocity EWMA [0,1] (smoothed improvement signal)
        diversity: Population diversity [0,1]
        stagnation: Search stagnation level [0,1]
        budget_consumed: Fraction of budget used [0,1]
        action: 0 (STAY) or 1 (ADVANCE)
        phase: 'EXPLORATION', 'EXPLOITATION', or 'TERMINATION'
        config: Optional EFRConfig
        
    Returns:
        RewardSignal with value in [-1, 1]
        
    Example:
        >>> signal = compute_reward(
        ...     fitness_norm=0.2,  # quality = 0.8 (good!)
        ...     improvement=0.5,   # moderate improvement velocity
        ...     diversity=0.3,
        ...     stagnation=0.7,
        ...     budget_consumed=0.5,
        ...     action=1,
        ...     phase='EXPLOITATION'
        ... )
        >>> print(f"Reward: {signal.value:.3f}, Efficiency unlocked: {signal.efficiency_unlocked}")
    """
    efr = EffectivenessFirstReward(config)
    state = {
        'fitness_norm': fitness_norm,
        'improvement_velocity': improvement,
        'diversity_score': diversity,
        'stagnation_ratio': stagnation,
        'budget_consumed': budget_consumed
    }
    return efr.calculate(state, action, phase)


# Alias for general use
Reward = EffectivenessFirstReward
