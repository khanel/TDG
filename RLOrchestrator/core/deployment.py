"""
Deployment wrapper for the trained orchestrator agent.

Provides configurable budget behavior at inference time WITHOUT retraining.

=== KEY INSIGHT ===
The agent decides to switch phases based on OBSERVATION signals (especially stagnation).
At inference time, we can SCALE these signals to influence behavior without touching the agent.

This is "observation shaping" - the agent still makes its own decisions,
but we change what it "sees" about the search state.

=== APPROACHES FOR CONTROLLING EXPLORATION/EXPLOITATION ===

1. STAGNATION SCALING (implemented):
   - Scale down stagnation signal → agent thinks search is still progressing
   - Lower stagnation = less likely to switch phases
   - stagnation_scale=0.5 means agent sees half the actual stagnation
   
2. STAGNATION CAPPING (alternative - TODO):
   - Cap stagnation at a maximum value (e.g., 0.6)
   - Agent never sees "very high" stagnation
   - cap_stagnation=0.6 means stagnation never exceeds 0.6
   
3. DELAYED STAGNATION (alternative - TODO):
   - Only allow stagnation to grow after X% of budget used
   - Before threshold: stagnation = 0
   - delay_stagnation_until=0.3 means no stagnation signal for first 30% of budget

4. IMPROVEMENT BOOST (alternative - TODO):
   - Artificially boost improvement signal to keep agent exploring
   - improvement_boost=0.2 adds 0.2 to improvement signal

5. ACTION MASKING (previous approach - less elegant):
   - Block certain actions directly
   - Overrides agent's decisions rather than influencing them
"""

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass
from stable_baselines3 import PPO


@dataclass 
class ObservationConfig:
    """Configuration for observation shaping at deployment time.
    
    These parameters modify what the agent "sees" without changing its policy.
    The agent still makes its own decisions based on the (shaped) observations.
    
    Primary control: stagnation_scale
    - 1.0 = no change (agent sees true stagnation)
    - 0.5 = agent sees half the stagnation → explores longer
    - 0.0 = agent never sees stagnation → explores until budget exhausted
    
    Future options (not yet implemented):
    - cap_stagnation: Maximum stagnation value agent can see
    - delay_stagnation_until: Budget fraction before stagnation activates
    - improvement_boost: Artificial boost to improvement signal
    """
    
    # Scale factor for stagnation signal (0.0 to 1.0)
    # Lower = agent sees less stagnation = explores longer
    stagnation_scale: float = 1.0
    
    # TODO: Cap stagnation at this value (None = no cap)
    # cap_stagnation: Optional[float] = None
    
    # TODO: Don't report stagnation until this fraction of budget used
    # delay_stagnation_until: float = 0.0
    
    # TODO: Add this value to improvement signal
    # improvement_boost: float = 0.0


# Preset configurations for common use cases
PRESETS = {
    "fast": ObservationConfig(
        stagnation_scale=2.0,  # Higher stagnation = switch faster
    ),
    "balanced": ObservationConfig(
        stagnation_scale=1.0,  # No modification
    ),
    "thorough": ObservationConfig(
        stagnation_scale=0.5,  # Half stagnation = explore longer
    ),
    "max_explore": ObservationConfig(
        stagnation_scale=0.2,  # Very low stagnation = maximize exploration
    ),
}


class DeployableOrchestrator:
    """
    Wrapper around trained PPO agent with observation shaping for deployment.
    
    === KEY CONCEPT ===
    Instead of masking actions or forcing ratios, we SHAPE the observations
    the agent sees. The agent still makes its own decisions, but based on
    modified signals.
    
    Primary mechanism: Stagnation scaling
    - Agent learned: "high stagnation → switch phase"
    - We scale stagnation down → agent thinks search is still progressing
    - Result: Agent naturally explores longer without us overriding its decisions
    
    Usage:
        agent = DeployableOrchestrator("model.zip", mode="thorough")
        action = agent.predict(observation)
    
    Modes:
        - "fast": 2x stagnation → switch phases quickly
        - "balanced": 1x stagnation → no modification (trained behavior)
        - "thorough": 0.5x stagnation → explore longer
        - "max_explore": 0.2x stagnation → maximize exploration time
    """
    
    # Observation indices (must match ObservationComputer)
    OBS_BUDGET = 0
    OBS_FITNESS = 1
    OBS_IMPROVEMENT_VELOCITY = 2  # Now EWMA of improvement velocity
    OBS_STAGNATION = 3
    OBS_DIVERSITY = 4
    OBS_PHASE = 5
    
    def __init__(
        self, 
        model_path: str,
        mode: Literal["fast", "balanced", "thorough", "max_explore"] = "balanced",
        custom_config: Optional[ObservationConfig] = None,
    ):
        self.model = PPO.load(model_path)
        
        if custom_config is not None:
            self.config = custom_config
        else:
            self.config = PRESETS.get(mode, PRESETS["balanced"])
        
        self.mode = mode
    
    def shape_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply observation shaping to influence agent behavior.
        
        Current implementation: Scale stagnation signal
        - Lower stagnation = agent thinks search is progressing = stays in current phase longer
        
        Args:
            obs: Raw observation [budget, fitness, improvement, stagnation, diversity, phase]
            
        Returns:
            Shaped observation with modified signals
        """
        shaped = obs.copy()
        
        # Scale stagnation signal
        # Lower scale = agent sees less stagnation = explores longer
        shaped[self.OBS_STAGNATION] = np.clip(
            obs[self.OBS_STAGNATION] * self.config.stagnation_scale,
            0.0, 1.0
        )
        
        # TODO: Implement other shaping methods
        # if self.config.cap_stagnation is not None:
        #     shaped[self.OBS_STAGNATION] = min(shaped[self.OBS_STAGNATION], self.config.cap_stagnation)
        
        # if self.config.delay_stagnation_until > 0:
        #     if obs[self.OBS_BUDGET] < self.config.delay_stagnation_until:
        #         shaped[self.OBS_STAGNATION] = 0.0
        
        # if self.config.improvement_boost > 0:
        #     shaped[self.OBS_IMPROVEMENT] = min(1.0, obs[self.OBS_IMPROVEMENT] + self.config.improvement_boost)
        
        return shaped
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action using shaped observation.
        
        The agent sees modified signals (especially stagnation) which influences
        its decisions without us directly overriding them.
        
        Args:
            obs: Raw observation from environment
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: 0 (STAY) or 1 (ADVANCE)
        """
        # Shape the observation before passing to agent
        shaped_obs = self.shape_observation(obs)
        
        # Agent makes decision based on shaped observation
        action, _ = self.model.predict(shaped_obs, deterministic=deterministic)
        
        return int(action)
    
    def get_config_summary(self) -> str:
        """Return human-readable config summary."""
        return (
            f"Mode: {self.mode}\n"
            f"  Stagnation scale: {self.config.stagnation_scale}x\n"
            f"  (lower = explore longer, higher = switch faster)"
        )


def create_deployable_agent(
    model_path: str,
    stagnation_scale: float = 1.0,
) -> DeployableOrchestrator:
    """
    Factory function to create a deployable agent with custom observation shaping.
    
    Args:
        model_path: Path to trained PPO model
        stagnation_scale: Scale factor for stagnation signal (0.0 to 2.0+)
            - 1.0 = no change (trained behavior)
            - 0.5 = agent sees half stagnation → explores longer
            - 0.2 = agent sees very low stagnation → maximizes exploration
            - 2.0 = agent sees double stagnation → switches phases quickly
        
    Returns:
        Configured DeployableOrchestrator
    """
    config = ObservationConfig(stagnation_scale=stagnation_scale)
    return DeployableOrchestrator(model_path, custom_config=config)
