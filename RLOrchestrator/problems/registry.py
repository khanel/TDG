"""
Registry for problem adapters (kept for backward compatibility).
"""

from importlib import import_module
from typing import Dict, Type, Any

_problem_registry: Dict[str, str] = {
    "tsp": "RLOrchestrator.tsp.adapter:TSPAdapter",
    "knapsack": "RLOrchestrator.knapsack.adapter:KnapsackAdapter",
    "maxcut": "RLOrchestrator.maxcut.adapter:MaxCutAdapter",
}


def _resolve(import_path: str) -> Type[Any]:
    module_name, attr = import_path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr)


def get_problem_adapter(name: str) -> Type[Any]:
    target = _problem_registry.get(name)
    if target is None:
        return None
    return _resolve(target)


def get_problem_registry() -> Dict[str, Type[Any]]:
    return {name: _resolve(path) for name, path in _problem_registry.items()}
