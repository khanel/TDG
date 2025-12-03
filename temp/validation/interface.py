from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RewardSignal:
    """
    Standardized return object for all reward calculations.
    Ensures tests can inspect *why* a reward was given, not just the value.
    """
    total_value: float          # The final scalar for the DRL agent (Bounded [-1, 1])
    raw_components: Dict[str, float]  # Breakdown: {'progress': 0.8, 'stagnation_penalty': -0.2}
    is_clamped: bool            # Flag if the value exceeded bounds and was clipped
    metadata: Dict[str, Any]    # Debug info: {'phase_weight_applied': 0.5}

class AbstractSearchReward(ABC):
    """
    The rigid standard for Search Mechanism Reward Functions.
    """
    # Hard Constraints for Normalization
    MIN_REWARD: float = -1.0
    MAX_REWARD: float = 1.0

    @abstractmethod
    def calculate(self, state_vector: dict, action: int, context: dict) -> RewardSignal:
        """
        Must implement the reward logic.
        
        Args:
            state_vector: Dictionary mapping to PDF parameters:
                          - 'fitness_norm': float (0-1)
                          - 'improvement_binary': int (0/1)
                          - 'diversity_score': float (0-1)
                          - 'stagnation_ratio': float (0-1)
                          - 'budget_consumed': float (0-1)
            action: 0 (Stay/Repeat), 1 (Advance/Transition)
            context: External context {'current_phase': 'EXPLORATION', 'global_best_found': bool}
            
        Returns:
            RewardSignal object.
        """
        pass

    def validate_bounds(self, value: float) -> float:
        """Enforces clipping."""
        return max(self.MIN_REWARD, min(self.MAX_REWARD, value))
