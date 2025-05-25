from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Generic, Optional, TypeVar, Union, Tuple, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Type variables for generic typing
StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')


class BaseEnvironment(gym.Env, Generic[StateType, ActionType]):
    """
    Abstract base class for RL environments compatible with Gymnasium.
    
    This is a generic class that can be specialized for different types of
    states and actions. The generic types are:
    - StateType: The type of the state representation
    - ActionType: The type of actions
    
    Subclasses should implement the core environment logic while maintaining
    this interface.
    """
    
    # Set these in subclasses
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            config: Configuration dictionary for the environment.
            render_mode: The render mode to use ('human', 'rgb_array', None).
        """
        super().__init__()
        self.config = config or {}
        self.render_mode = render_mode
        self.current_step = 0
        self.max_steps = self.config.get('max_steps', 1000)
        self._state = None
        
        # Initialize spaces (must be set in subclasses)
        self.action_space = None  # Must be set in subclass
        self.observation_space = None  # Must be set in subclass
    
    @property
    def state(self) -> StateType:
        """Get the current state of the environment."""
        return self._state
    
    @abstractmethod
    def reset(self, 
             *, 
             seed: Optional[int] = None, 
             options: Optional[Dict[str, Any]] = None
    ) -> Tuple[StateType, Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Optional seed for the random number generator.
            options: Additional options for the reset process.
            
        Returns:
            Tuple of (initial_observation, info).
        """
        super().reset(seed=seed)
        if options is None:
            options = {}
        self.current_step = 0
        return self._state, {}
    
    @abstractmethod
    def step(self, action: ActionType) -> Tuple[StateType, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take.
            
        Returns:
            Tuple containing:
            - observation: The new state
            - reward: The reward for the action
            - terminated: Whether the episode has ended
            - truncated: Whether the episode was truncated (e.g., due to time limit)
            - info: Additional information
        """
        self.current_step += 1
        return self._state, 0.0, False, False, {}
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            The rendered frame if applicable, None otherwise.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def _render_frame(self) -> np.ndarray:
        """
        Render a single frame of the environment.
        
        Returns:
            RGB frame as a numpy array.
        """
        raise NotImplementedError("Rendering not implemented")
    
    def close(self) -> None:
        """Perform any necessary cleanup."""
        pass
    
    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set the random seed for the environment.
        
        Args:
            seed: The random seed to use.
        """
        super().reset(seed=seed)
        np.random.seed(seed)
