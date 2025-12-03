from typing import Dict, Any
from interface import AbstractSearchReward, RewardSignal

class HybridSearchReward(AbstractSearchReward):
    """
    Concrete implementation of the Explore -> Exploit -> Terminate reward logic.
    """

    def calculate(self, state_vector: dict, action: int, context: dict) -> RewardSignal:
        # Extract inputs
        improvement = state_vector.get('improvement_binary', 0)
        diversity = state_vector.get('diversity_score', 0.0)
        stagnation = state_vector.get('stagnation_ratio', 0.0)
        budget = state_vector.get('budget_consumed', 0.0)
        fitness_norm = state_vector.get('fitness_norm', 0.0)
        
        phase = context.get('current_phase', 'EXPLORATION')
        
        components = {}
        total_reward = 0.0
        
        # --- Global Budget Constraints (Exclusive Overrides) ---
        # B3: The Deadline (Budget Critical -> Must Terminate)
        if budget > 0.98 and action == 1:
            r = 0.5
            components['deadline_compliance'] = r
            total_reward += r

        # B2: Infinite Budget Illusion (High budget, quitting early)
        elif budget < 0.1 and action == 1 and improvement == 1:
             r = -0.8
             components['premature_budget_exit'] = r
             total_reward += r

        # E7: Nervous Starter (Switching immediately)
        elif budget < 0.05 and action == 1 and phase == 'EXPLORATION':
            r = -0.5
            components['nervous_starter_penalty'] = r
            total_reward += r

        else:
            # --- Phase Specific Logic ---
            if phase == 'EXPLORATION':
                if action == 0: # Continue (Stay in Exploration)
                    # E1: Eureka (Improvement + Diversity)
                    if improvement == 1:
                        # Reward finding new peaks, weighted by diversity
                        r = 0.2 + (0.3 * diversity)
                        components['base_reward'] = 0.2
                        components['diversity_bonus'] = 0.3 * diversity
                        total_reward += r
                    
                    # E2: Blind Walk (No improvement, but high diversity)
                    elif diversity > 0.7 and stagnation < 0.5:
                        r = 0.1
                        components['diversity_maintenance'] = r
                        total_reward += r
                    
                    # E3: Local Trap (Stagnant and not improving)
                    elif stagnation > 0.5:
                        # Ramp from -0.05 to -0.5
                        penalty_strength = (stagnation - 0.5) / 0.5
                        base_penalty = -0.05 - (0.45 * penalty_strength)
                        
                        # Diversity Mitigation
                        mitigation_factor = 1.0 - (diversity * 0.8)
                        r = base_penalty * mitigation_factor
                        
                        components['stagnation_penalty'] = r
                        total_reward += r
                    
                    else:
                        # Neutral/Small penalty for wasting time without result
                        total_reward += -0.05

                    # *** B1: Panic Mode (Additive) ***
                    # Smoothly apply penalty as budget exceeds 80%
                    if budget > 0.8:
                        penalty_strength = (budget - 0.8) / 0.2
                        r = -0.8 * penalty_strength
                        components['budget_panic_penalty'] = r
                        total_reward += r
                        
                elif action == 1: # Switch (to Exploitation)
                    # E4: Timely Escape (High stagnation -> Switch)
                    if stagnation > 0.8:
                        r = 0.5
                        components['transition_reward'] = r
                        total_reward += r
                    
                    # E5: Premature Exit (Leaving while good)
                    elif improvement == 1 or diversity > 0.7:
                        r = -0.8
                        components['premature_penalty'] = r
                        total_reward += r
                    
                    else:
                        # Neutral transition
                        total_reward += 0.0

            elif phase == 'EXPLOITATION':
                if action == 0: # Continue (Stay in Exploitation)
                    # X1: Gold Mine (Improvement)
                    if improvement == 1:
                        # Higher weight for exploitation improvements than exploration
                        r = 0.5 + (0.5 * fitness_norm) 
                        components['improvement_reward'] = r
                        total_reward += r
                    
                    # X3: Dead End (Stagnant)
                    elif stagnation > 0.8:
                        r = -0.5
                        components['stagnation_penalty'] = r
                        total_reward += r
                    
                    else:
                        # X2: The Grind (Small positive/neutral if not stagnant)
                        if stagnation < 0.5:
                            r = 0.05
                            components['grind_bonus'] = r
                            total_reward += r
                        else:
                            # Slight penalty for dragging on
                            total_reward += -0.1

                elif action == 1: # Switch (Terminate)
                    # X4: Victory Lap (Converged -> Terminate)
                    if stagnation > 0.9:
                        r = 0.5
                        components['termination_reward'] = r
                        total_reward += r
                    
                    # X5: Quitter (Leaving while improving)
                    elif improvement == 1 or stagnation < 0.3:
                        r = -0.8
                        components['premature_penalty'] = r
                        total_reward += r
                    
                    else:
                        total_reward += 0.0

        # Clamp result to [-1, 1]
        final_value = self.validate_bounds(total_reward)
        is_clamped = (final_value != total_reward)
        
        return RewardSignal(
            total_value=final_value,
            raw_components=components,
            is_clamped=is_clamped,
            metadata={'phase': phase, 'action': action, 'raw_total': total_reward}
        )
