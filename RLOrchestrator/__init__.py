"""
RL Orchestrator Framework (Core-compatible)

Lightweight orchestration and RL environment tools built on top of the
root Core/ APIs (`Core.problem` and `Core.search_algorithm`).
"""

from .core.orchestrator import OrchestratorEnv
from .core.observation import ObservationComputer
from .core.env_factory import create_env
from .problems import get_problem_registry
from .solvers import get_solver_registry

__version__ = "0.1.0"
__all__ = [
    'OrchestratorEnv', 'create_env', 'ObservationComputer',
    'get_problem_registry', 'get_solver_registry',
    '__version__'
]
