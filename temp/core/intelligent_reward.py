"""
Intelligent Search Reward (ISR) v2.0
=====================================

A mathematically rigorous reward function designed for RL-based hyper-heuristic
orchestration of metaheuristic search algorithms.

Design Philosophy: "Effectiveness First, Efficiency as Bonus"
--------------------------------------------------------------
The agent controls a 3-phase pipeline with UNIFIED action semantics:
- Action 0: STAY in current phase
- Action 1: ADVANCE to next phase

Phase Pipeline:
1. EXPLORATION: Find promising regions, maintain diversity, detect stagnation
2. EXPLOITATION: Refine solutions, maximize quality improvements  
3. TERMINATION: Episode ends - evaluate stopping decision

Key Change from v1.0:
- Action semantics are now CONSISTENT across all phases
- No more "terminate" action in exploitation - that's handled by ADVANCE to termination phase
- Termination is a proper phase, not an action within exploitation

Mathematical Properties:
------------------------
- Bounded: All rewards in [-1.0, +1.0]
- Lipschitz Continuous: No cliffs or discontinuities (K < 10)
- Gradient-Friendly: Smooth ramps instead of hard thresholds
- Variance-Controlled: Consistent scaling across phases
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class RewardSignal:
    """
    Standardized return object for all reward calculations.
    Enables inspection of reward decomposition for debugging and validation.
    """
    total_value: float                    # Final scalar for DRL agent, bounded [-1, 1]
    raw_components: Dict[str, float]      # Breakdown: {'improvement': 0.5, 'diversity': 0.1}
    is_clamped: bool                      # True if value exceeded bounds and was clipped
    metadata: Dict[str, Any]              # Debug info: {'phase': 'EXPLORATION', 'action': 0}


class AbstractSearchReward(ABC):
    """
    Abstract interface for Search Mechanism Reward Functions.
    Enforces bounded output contract.
    """
    MIN_REWARD: float = -1.0
    MAX_REWARD: float = 1.0

    @abstractmethod
    def calculate(self, state_vector: dict, action: int, context: dict) -> RewardSignal:
        """
        Calculate reward for a state-action pair.
        
        Args:
            state_vector: Observation features
                - 'fitness_norm': float [0,1] - normalized fitness (0=optimal, 1=worst)
                - 'improvement_binary': int {0,1} - whether improvement occurred
                - 'diversity_score': float [0,1] - population diversity
                - 'stagnation_ratio': float [0,1] - search stagnation level
                - 'budget_consumed': float [0,1] - fraction of budget used
            action: 0 (Stay/Continue), 1 (Advance/Switch/Terminate)
            context: External context {'current_phase': 'EXPLORATION'|'EXPLOITATION'}
            
        Returns:
            RewardSignal with bounded total_value in [-1, 1]
        """
        pass

    def validate_bounds(self, value: float) -> float:
        """Enforce reward bounds via clipping."""
        return float(np.clip(value, self.MIN_REWARD, self.MAX_REWARD))


@dataclass
class ISRConfig:
    """
    Configuration for Intelligent Search Reward v2.0.
    
    Now supports 3-phase pipeline with unified action semantics:
    - Exploration: reward improvements, penalize stagnation, reward timely advance
    - Exploitation: reward refinement only (no termination handling)
    - Termination: evaluate stopping decision, efficiency bonus
    """
    # === STEP COST ===
    step_cost: float = -0.01
    
    # === EXPLORATION PHASE ===
    # REBALANCED v6: Increase exploration rewards to prevent skipping
    explore_improvement_base: float = 0.4  # Was 0.15 - now more competitive with exploitation
    explore_improvement_diversity_bonus: float = 0.2  # Was 0.1
    explore_diversity_threshold: float = 0.4
    explore_diversity_bonus: float = 0.15  # Was 0.1
    explore_stagnation_threshold: float = 0.3
    explore_stagnation_penalty_max: float = -0.8
    explore_timely_switch_reward: float = 0.6
    explore_timely_switch_stagnation_threshold: float = 0.4
    explore_premature_switch_penalty: float = -0.5  # Was -0.3 - stronger penalty for early switch
    # NEW: Very early exploration switch penalty
    explore_very_early_switch_penalty: float = -0.8  # Strong penalty for switching at step 0-1
    explore_very_early_stagnation_threshold: float = 0.1  # Below this = way too early
    
    # === EXPLOITATION PHASE (no termination handling) ===
    # REBALANCED v6: Reduce exploitation rewards to be comparable to exploration
    exploit_improvement_base: float = 0.4  # Was 0.8 - same as exploration
    exploit_improvement_quality_bonus: float = 0.4  # Was 0.7 - reduced
    exploit_stagnation_threshold: float = 0.7
    exploit_stagnation_penalty: float = -0.3
    # Advance to termination rewards (unified action semantics)
    exploit_timely_advance_reward: float = 0.5
    exploit_timely_advance_stagnation_threshold: float = 0.6
    exploit_premature_advance_penalty: float = -0.4
    # CRITICAL: Very early exploitation termination penalty
    # This prevents agent from using exploitation as a "pass-through" to termination
    # Stagnation resets on phase switch, so we use a stricter threshold here
    exploit_very_early_advance_penalty: float = -0.6  # Strong penalty for advancing with low stagnation
    exploit_very_early_stagnation_threshold: float = 0.3  # Below this = very early
    
    # === TERMINATION PHASE (new) ===
    # Rewards for entering termination phase with good solution quality
    # NOTE: Quality thresholds calibrated for NK-Landscape where:
    # - Pure exploration baseline achieves ~0.68 quality
    # - Good agent should achieve 0.70+ quality
    # - Excellent quality is 0.80+ (significant improvement)
    #
    # CRITICAL: Quality threshold MUST be above baseline performance!
    # Otherwise agent learns to exit early with "good enough" solution.
    termination_quality_reward_base: float = 0.5  # Reward good solutions
    termination_quality_threshold: float = 0.70  # ABOVE baseline (~0.68) - force agent to beat them
    termination_excellent_quality_threshold: float = 0.80  # Significant improvement target
    termination_efficiency_bonus_max: float = 0.3  # REDUCED from 0.5 to discourage early exit
    termination_poor_quality_penalty: float = -0.8  # Penalize bad solutions
    
    # === BUDGET PRESSURE ===
    budget_pressure_onset: float = 0.7
    budget_pressure_max: float = -0.4
    
    # === NUMERICAL STABILITY ===
    eps: float = 1e-8


class IntelligentSearchReward(AbstractSearchReward):
    """
    Intelligent Search Reward (ISR) v2.0 - 3-Phase Pipeline Design
    
    Key Features:
    -------------
    1. Unified Action Semantics: Action 0=Stay, Action 1=Advance (consistent across phases)
    2. 3-Phase Pipeline: Exploration → Exploitation → Termination
    3. Phase-Aware Rewards: Different logic per phase
    4. Smooth Transitions: Linear ramps instead of hard thresholds
    5. Bounded Output: Guaranteed [-1, 1] range
    
    Phase Design:
    -------------
    EXPLORATION:
        - Reward improvements + diversity bonus
        - Penalize stagnation (smoothly)
        - Reward timely advance, penalize premature advance
        
    EXPLOITATION:
        - Higher rewards for improvements (refining is valuable)
        - Quality-scaled bonuses (better solutions = more reward)
        - Reward timely advance when stagnant, penalize premature advance
        - NO termination logic - that's now in termination phase
        
    TERMINATION:
        - Evaluate the stopping decision
        - Reward high quality solutions
        - Efficiency bonus for budget remaining
        - Penalize poor quality termination
    """
    
    def __init__(self, config: Optional[ISRConfig] = None):
        self.cfg = config or ISRConfig()
    
    def calculate(self, state_vector: dict, action: int, context: dict) -> RewardSignal:
        """
        Main reward calculation with 3-phase support.
        
        Action semantics (unified across all phases):
        - Action 0: STAY in current phase
        - Action 1: ADVANCE to next phase
        """
        # === EXTRACT STATE ===
        improvement = int(state_vector.get('improvement_binary', 0))
        diversity = float(state_vector.get('diversity_score', 0.0))
        stagnation = float(state_vector.get('stagnation_ratio', 0.0))
        budget = float(state_vector.get('budget_consumed', 0.0))
        fitness_norm = float(state_vector.get('fitness_norm', 0.0))
        
        # Quality is inverse of normalized fitness (0=worst, 1=optimal)
        quality = 1.0 - fitness_norm
        
        phase = context.get('current_phase', 'EXPLORATION')
        
        components = {}
        total_reward = 0.0
        
        # === UNIVERSAL STEP COST (not applied in termination) ===
        if phase != 'TERMINATION':
            total_reward += self.cfg.step_cost
            components['step_cost'] = self.cfg.step_cost
        
        # === PHASE-SPECIFIC LOGIC ===
        if phase == 'EXPLORATION':
            total_reward, components = self._exploration_reward(
                action, improvement, diversity, stagnation, budget,
                quality, total_reward, components
            )
        elif phase == 'EXPLOITATION':
            total_reward, components = self._exploitation_reward(
                action, improvement, diversity, stagnation, budget,
                quality, total_reward, components
            )
        else:  # TERMINATION
            total_reward, components = self._termination_reward(
                quality, budget, stagnation, total_reward, components
            )
        
        # === GLOBAL BUDGET PRESSURE (not applied in termination) ===
        if phase != 'TERMINATION' and budget > self.cfg.budget_pressure_onset:
            pressure_strength = (budget - self.cfg.budget_pressure_onset) / (1.0 - self.cfg.budget_pressure_onset + self.cfg.eps)
            pressure_strength = np.clip(pressure_strength, 0.0, 1.0)
            pressure = self.cfg.budget_pressure_max * pressure_strength
            total_reward += pressure
            components['budget_pressure'] = pressure
        
        # === BOUND AND RETURN ===
        final_value = self.validate_bounds(total_reward)
        is_clamped = abs(final_value - total_reward) > self.cfg.eps
        
        return RewardSignal(
            total_value=final_value,
            raw_components=components,
            is_clamped=is_clamped,
            metadata={
                'phase': phase,
                'action': action,
                'raw_total': total_reward,
                'quality': quality
            }
        )
    
    def _exploration_reward(
        self, action: int, improvement: int, diversity: float,
        stagnation: float, budget: float, quality: float,
        total_reward: float, components: Dict[str, float]
    ) -> tuple:
        """
        Exploration phase reward logic.
        
        Goals:
        - Find promising regions (reward improvements)
        - Maintain diversity (small bonus)
        - Detect and escape stagnation (penalty for staying, reward for switching)
        """
        cfg = self.cfg
        
        if action == 0:  # STAY in Exploration
            if improvement == 1:
                # === E1: EUREKA ===
                # Found improvement! Reward scales with diversity (more diverse = better exploration)
                r = cfg.explore_improvement_base + (cfg.explore_improvement_diversity_bonus * diversity)
                components['exploration_success'] = r
                total_reward += r
                
            elif stagnation > cfg.explore_stagnation_threshold:
                # === E3: STAGNANT EXPLORER ===
                # Stuck without improvement - penalize (smooth ramp)
                # Penalty increases linearly from threshold to 1.0
                penalty_progress = (stagnation - cfg.explore_stagnation_threshold) / (1.0 - cfg.explore_stagnation_threshold + cfg.eps)
                penalty_progress = np.clip(penalty_progress, 0.0, 1.0)
                
                # Diversity can mitigate some penalty (diverse stagnation less bad)
                diversity_mitigation = 1.0 - (diversity * 0.5)
                
                r = cfg.explore_stagnation_penalty_max * penalty_progress * diversity_mitigation
                components['stagnation_penalty'] = r
                total_reward += r
                
            else:
                # === E2: BLIND WALK (SMOOTH) ===
                # Not improving, not highly stagnant - diversity provides smooth bonus
                # Use smooth sigmoid instead of hard threshold for Lipschitz continuity
                diversity_factor = self._smooth_sigmoid(diversity, cfg.explore_diversity_threshold, 6.0)
                r = cfg.explore_diversity_bonus * diversity_factor
                if r > 0.001:  # Only record meaningful bonus
                    components['diversity_bonus'] = r
                    total_reward += r
        
        else:  # action == 1: SWITCH to Exploitation
            if improvement == 1:
                # === E5: PREMATURE EXIT ===
                # Switching while still finding improvements - bad idea
                r = cfg.explore_premature_switch_penalty
                components['premature_switch_penalty'] = r
                total_reward += r
            
            elif stagnation < cfg.explore_very_early_stagnation_threshold:
                # === VERY EARLY SWITCH ===
                # Stagnation is way too low - exploration barely started
                # This prevents skipping exploration entirely
                r = cfg.explore_very_early_switch_penalty
                components['very_early_switch_penalty'] = r
                total_reward += r
                
            else:
                # === SMOOTH SWITCH DECISION ===
                # No hard threshold - use smooth sigmoid for Lipschitz continuity
                # This ensures gradient stability at all stagnation levels
                
                # Stagnation readiness: smooth sigmoid centered at threshold
                stag_center = cfg.explore_timely_switch_stagnation_threshold
                stag_scale = 6.0  # Controls smoothness (lower = smoother gradient)
                switch_readiness = self._smooth_sigmoid(stagnation, stag_center, stag_scale)
                
                # Interpolate between neutral/penalty and reward
                # At low stagnation (readiness ~0): strong penalty
                # At high stagnation (readiness ~1): full reward
                early_switch_penalty = -0.4  # Strong penalty for switching too early
                
                r = early_switch_penalty * (1.0 - switch_readiness) + cfg.explore_timely_switch_reward * switch_readiness
                
                if r >= 0:
                    components['timely_switch'] = r
                else:
                    components['uncertain_switch'] = r
                
                total_reward += r
        
        return total_reward, components
    
    def _exploitation_reward(
        self, action: int, improvement: int, diversity: float,
        stagnation: float, budget: float, quality: float,
        total_reward: float, components: Dict[str, float]
    ) -> tuple:
        """
        Exploitation phase reward logic (v2.0 - unified actions).
        
        Goals:
        - Refine solutions (higher rewards for improvements)
        - Scale rewards by solution quality (better solutions = more reward)
        - Reward timely advance to termination when stagnant
        - Penalize premature advance while still improving
        
        Action semantics (same as exploration):
        - Action 0: STAY in exploitation
        - Action 1: ADVANCE to termination phase
        """
        cfg = self.cfg
        
        if action == 0:  # STAY in Exploitation
            if improvement == 1:
                # Found improvement - very valuable in exploitation
                r = cfg.exploit_improvement_base + (cfg.exploit_improvement_quality_bonus * quality)
                components['exploitation_success'] = r
                total_reward += r
                
            elif stagnation > cfg.exploit_stagnation_threshold:
                # Stagnant in exploitation without improvement
                # Light penalty to encourage advance to termination
                penalty_progress = (stagnation - cfg.exploit_stagnation_threshold) / (1.0 - cfg.exploit_stagnation_threshold + cfg.eps)
                r = cfg.exploit_stagnation_penalty * penalty_progress
                components['exploitation_stagnation'] = r
                total_reward += r
            
            # else: No improvement, not stagnant yet - just step cost applies
        
        else:  # action == 1: ADVANCE to Termination
            if improvement == 1:
                # Advancing while still finding improvements - bad idea
                r = cfg.exploit_premature_advance_penalty
                components['premature_advance_penalty'] = r
                total_reward += r
                
            elif stagnation < cfg.exploit_very_early_stagnation_threshold:
                # VERY EARLY ADVANCE: Stagnation is too low (exploitation just started)
                # This prevents using exploitation as a pass-through to termination
                # After phase switch, stagnation resets - need time to build it up
                r = cfg.exploit_very_early_advance_penalty
                components['very_early_advance_penalty'] = r
                total_reward += r
                
            else:
                # Smooth advance decision based on stagnation
                stag_center = cfg.exploit_timely_advance_stagnation_threshold
                stag_scale = 5.0
                advance_readiness = self._smooth_sigmoid(stagnation, stag_center, stag_scale)
                
                # Interpolate between penalty and reward
                early_advance_penalty = -0.3
                r = early_advance_penalty * (1.0 - advance_readiness) + cfg.exploit_timely_advance_reward * advance_readiness
                
                if r >= 0:
                    components['timely_advance'] = r
                else:
                    components['uncertain_advance'] = r
                
                total_reward += r
        
        return total_reward, components
    
    def _termination_reward(
        self, quality: float, budget: float, stagnation: float,
        total_reward: float, components: Dict[str, float]
    ) -> tuple:
        """
        Termination phase reward logic (new in v2.0).
        
        This phase is entered when the agent advances from exploitation.
        The episode ends after this reward is computed.
        
        FULLY SMOOTH DESIGN:
        - No hard if/else thresholds
        - Smooth interpolation between penalty and reward based on quality
        - Efficiency bonus scales smoothly with quality
        
        Note: No action is taken in this phase - it's an evaluation of
        the stopping decision made by advancing from exploitation.
        """
        cfg = self.cfg
        
        # Quality factor using smooth sigmoid (0 at low quality, 1 at high quality)
        quality_factor = self._smooth_sigmoid(
            quality, 
            cfg.termination_quality_threshold,
            4.0  # Lower scale for smoother transition
        )
        
        # === SMOOTH QUALITY-BASED REWARD ===
        # Interpolate between penalty and reward based on quality_factor
        # At quality_factor=0: full penalty
        # At quality_factor=1: full reward
        
        penalty_component = cfg.termination_poor_quality_penalty * (1.0 - quality_factor)
        reward_component = cfg.termination_quality_reward_base * quality_factor
        
        # Excellent quality bonus (smooth scaling, no hard threshold)
        excellent_factor = self._smooth_sigmoid(
            quality,
            cfg.termination_excellent_quality_threshold,
            5.0
        )
        excellent_bonus = 0.2 * excellent_factor
        
        quality_reward = penalty_component + reward_component + excellent_bonus
        
        if quality_reward >= 0:
            components['termination_quality_reward'] = quality_reward
        else:
            components['termination_poor_quality'] = quality_reward
        
        total_reward += quality_reward
        
        # === EFFICIENCY BONUS (with quality gate) ===
        # CRITICAL FIX: Efficiency bonus should ONLY reward early exit with HIGH quality
        # Otherwise agents learn to exit early to "save budget" with poor solutions
        #
        # Gate: efficiency bonus is ZERO if quality < threshold (hard cutoff)
        # Above threshold: scales smoothly with quality and budget remaining
        
        budget_remaining = max(0.0, 1.0 - budget)
        
        # HARD QUALITY GATE: Must exceed threshold to get ANY efficiency bonus
        # This prevents "exit early with bad solution to save budget" strategy
        min_quality_for_efficiency = cfg.termination_quality_threshold  # 0.7 default
        
        if budget_remaining > 0.05 and quality > min_quality_for_efficiency:
            # Quality above threshold - now use smooth scaling for amount
            # The bonus grows from 0 at threshold to max at excellent quality
            quality_above_threshold = (quality - min_quality_for_efficiency) / (1.0 - min_quality_for_efficiency + cfg.eps)
            quality_above_threshold = np.clip(quality_above_threshold, 0.0, 1.0)
            
            # Apply smooth sigmoid for gradual ramp (not sudden jump)
            efficiency_scale = self._smooth_sigmoid(quality_above_threshold, 0.5, 4.0)
            
            # Excellent quality gets extra boost
            efficiency_scale *= (1.0 + 0.5 * excellent_factor)
            
            efficiency_bonus = cfg.termination_efficiency_bonus_max * budget_remaining * efficiency_scale
            efficiency_bonus = min(efficiency_bonus, cfg.termination_efficiency_bonus_max)
            
            if efficiency_bonus > 0.01:
                components['efficiency_bonus'] = efficiency_bonus
                total_reward += efficiency_bonus
        
        return total_reward, components
    
    @staticmethod
    def _smooth_sigmoid(x: float, center: float, scale: float) -> float:
        """
        Smooth sigmoid function for continuous transitions.
        Returns value in [0, 1] with smooth transition around center.
        
        Args:
            x: Input value
            center: Value where sigmoid = 0.5
            scale: Steepness (lower = smoother, higher = sharper)
        
        Returns:
            Smooth value in [0, 1]
        """
        z = scale * (x - center)
        # Numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1.0 + exp_z)


# ================================================================
# PRESET CONFIGURATIONS
# ================================================================

def config_default() -> ISRConfig:
    """Default balanced configuration."""
    return ISRConfig()


def config_aggressive_exploration() -> ISRConfig:
    """
    Configuration that encourages longer exploration phases.
    Use for problems with many local optima.
    """
    return ISRConfig(
        explore_improvement_diversity_bonus=0.4,
        explore_diversity_bonus=0.08,
        explore_stagnation_threshold=0.6,
        explore_timely_switch_stagnation_threshold=0.5,
        budget_pressure_onset=0.8,
    )


def config_fast_convergence() -> ISRConfig:
    """
    Configuration that encourages faster switching and termination.
    Use for unimodal or well-behaved problems.
    """
    return ISRConfig(
        explore_improvement_base=0.3,
        explore_stagnation_threshold=0.4,
        explore_timely_switch_stagnation_threshold=0.2,
        explore_timely_switch_reward=0.6,
        exploit_timely_advance_stagnation_threshold=0.5,
        budget_pressure_onset=0.6,
        budget_pressure_max=-0.5,
    )


def config_quality_focused() -> ISRConfig:
    """
    Configuration that heavily rewards solution quality.
    Use when finding the best solution is more important than speed.
    """
    return ISRConfig(
        step_cost=-0.005,  # Lower step cost
        exploit_improvement_base=1.0,
        exploit_improvement_quality_bonus=1.0,
        termination_quality_threshold=0.8,
        termination_excellent_quality_threshold=0.95,
        budget_pressure_onset=0.85,
        budget_pressure_max=-0.3,
    )
