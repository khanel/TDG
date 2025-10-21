"""
Registry for external solver classes (Core-compatible).
Maps solver names to their SearchAlgorithm classes (e.g., from TSP/solvers/ or GWO/).
"""

from typing import Dict, Type, Any
from Core.search_algorithm import SearchAlgorithm


_solver_registry: Dict[str, Dict[str, Type[SearchAlgorithm]]] = {
    "exploration": {},
    "exploitation": {},
}


def register_exploration_solver(name: str, solver_class: Type[SearchAlgorithm]) -> None:
    """Register an exploration solver class."""
    _solver_registry["exploration"][name] = solver_class


def register_exploitation_solver(name: str, solver_class: Type[SearchAlgorithm]) -> None:
    """Register an exploitation solver class."""
    _solver_registry["exploitation"][name] = solver_class


def get_exploration_solver(name: str) -> Type[SearchAlgorithm]:
    """Get an exploration solver class by name."""
    return _solver_registry["exploration"].get(name)


def get_exploitation_solver(name: str) -> Type[SearchAlgorithm]:
    """Get an exploitation solver class by name."""
    return _solver_registry["exploitation"].get(name)


def get_solver_registry() -> Dict[str, Dict[str, Type[SearchAlgorithm]]]:
    """Return the full solver registry."""
    return _solver_registry.copy()


# Example registrations (external solvers should call these)
# register_exploration_solver("map_elites", TSPMapElitesSolver)
# register_exploitation_solver("sa", SimulatedAnnealing)
# register_exploitation_solver("gwo", GrayWolfOptimization)
