"""
Registry for discovering and categorizing solver classes.
"""

from typing import Dict, List, Type

from Core.search_algorithm import SearchAlgorithm

from ..problems.registry import list_problem_definitions, ProblemDefinition, SolverFactory

_solver_registry: Dict[str, Dict[str, List[Type[SearchAlgorithm]]]] = {}


def _discover_solvers():
    """Load solver classes directly from registered ProblemDefinitions."""
    if _solver_registry:
        return

    definitions = list_problem_definitions()
    for problem_name, definition in definitions.items():
        stage_map: Dict[str, List[Type[SearchAlgorithm]]] = {
            "exploration": [],
            "exploitation": [],
        }
        for phase, factory in (definition.solvers or {}).items():
            cls = factory.cls if isinstance(factory, SolverFactory) else None
            if not cls or not issubclass(cls, SearchAlgorithm):
                continue
            stage_map.setdefault(phase, []).append(cls)
        _solver_registry[problem_name] = stage_map

def get_solver_registry() -> Dict[str, Dict[str, List[Type[SearchAlgorithm]]]]:
    """Return the full solver registry, discovering solvers if not already done."""
    _discover_solvers()
    return _solver_registry

def get_solvers(problem: str, phase: str) -> List[Type[SearchAlgorithm]]:
    """
    Get a list of available solver classes for a given problem and phase.
    """
    registry = get_solver_registry()
    return registry.get(problem, {}).get(phase, [])
