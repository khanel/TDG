from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
import numpy as np

# Type variables for generic typing
StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')


class StrategyType(Enum):
    """Type of strategy (exploration or exploitation)."""
    EXPLORATION = auto()
    EXPLOITATION = auto()


@dataclass
class StrategyState(Generic[StateType]):
    """Container for strategy state."""
    state: StateType
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class StrategyAction(Generic[ActionType]):
    """Container for strategy action."""
    action: ActionType
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class BaseStrategy(ABC, Generic[StateType, ActionType]):
    """
    Abstract base class for search strategies.
    
    This is a generic class that can be specialized for different types of
    states and actions. The generic types are:
    - StateType: The type of the state representation
    - ActionType: The type of actions
    
    Subclasses should implement the core strategy logic while maintaining
    this interface.
    """
    
    def __init__(self, 
                 strategy_type: StrategyType,
                 config: Dict[str, Any] = None):
        """
        Initialize the base strategy.
        
        Args:
            strategy_type: Type of strategy (exploration or exploitation).
            config: Configuration dictionary for the strategy.
        """
        self.strategy_type = strategy_type
        self.config = config or {}
        self.current_step = 0
        self.metrics = {}
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the strategy with problem-specific parameters.
        
        Args:
            **kwargs: Strategy-specific initialization parameters.
        """
        pass
    
    @abstractmethod
    def get_action(self, state: StateType) -> StrategyAction[ActionType]:
        """
        Get the next action to take based on the current state.
        
        Args:
            state: Current state of the environment.
            
        Returns:
            StrategyAction containing the action and parameters.
        """
        pass
    
    @abstractmethod
    def update(self, 
              state: StateType, 
              action: StrategyAction[ActionType],
              next_state: StateType,
              reward: float,
              done: bool,
              **kwargs) -> Dict[str, Any]:
        """
        Update the strategy based on the observed transition.
        
        Args:
            state: Current state.
            action: Action taken.
            next_state: Next state.
            reward: Reward received.
            done: Whether the episode is done.
            **kwargs: Additional information.
            
        Returns:
            Dictionary containing any metrics or information to log.
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the strategy to its initial state.
        """
        self.current_step = 0
        self.metrics = {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics from the strategy.
        
        Returns:
            Dictionary containing strategy metrics.
        """
        return self.metrics
    
    def is_done(self) -> bool:
        """
        Check if the strategy has completed its execution.
        
        Returns:
            bool: True if the strategy is done, False otherwise.
        """
        return False
