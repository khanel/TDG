"""
Registry for problem adapters.
"""

from typing import Dict, Type, Any
from .tsp import TSPAdapter
from .knapsack import KnapsackAdapter


_problem_registry: Dict[str, Type[Any]] = {
    "tsp": TSPAdapter,
    "knapsack": KnapsackAdapter,
}


def get_problem_adapter(name: str) -> Type[Any]:
    """Get a problem adapter class by name."""
    return _problem_registry.get(name)


def get_problem_registry() -> Dict[str, Type[Any]]:
    """Return the full problem registry."""
    return _problem_registry.copy()