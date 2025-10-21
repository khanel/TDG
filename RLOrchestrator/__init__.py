"""
RL Orchestrator Framework (Core-compatible)

Lightweight orchestration and RL environment tools built on top of the
root Core/ APIs (`Core.problem` and `Core.search_algorithm`).
"""

from .core import Orchestrator, ObservationComputer, RewardComputer
from .problems import get_problem_registry
from .solvers import get_solver_registry

__version__ = "0.1.0"
__all__ = [
    'Orchestrator', 'ObservationComputer', 'RewardComputer',
    'get_problem_registry', 'get_solver_registry',
    '__version__'
]
