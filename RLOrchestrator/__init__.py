"""
RL Orchestrator Framework

A general-purpose reinforcement learning framework for orchestrating between
exploration and exploitation strategies in optimization problems.
"""

from .core import (
    BaseEnvironment, EnvironmentState, EnvironmentAction,
    BaseStrategy, StrategyType, StrategyState, StrategyAction,
    BaseAgent, AgentConfig
)

__version__ = "0.1.0"
__all__ = [
    # Core components
    'BaseEnvironment', 'EnvironmentState', 'EnvironmentAction',
    'BaseStrategy', 'StrategyType', 'StrategyState', 'StrategyAction',
    'BaseAgent', 'AgentConfig',
    
    # Version
    '__version__'
]
