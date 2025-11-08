"""
Registry for discovering and categorizing solver classes.
"""

import pkgutil
import importlib
from typing import Dict, List, Type
from Core.search_algorithm import SearchAlgorithm

_solver_registry: Dict[str, Dict[str, List[Type[SearchAlgorithm]]]] = {}

def _discover_solvers():
    """Dynamically discover solvers from RLOrchestrator subpackages."""
    if _solver_registry:
        return

    import RLOrchestrator
    package = RLOrchestrator
    
    for _, package_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if ".solvers" in package_name:
            continue # Avoid double-importing

        solvers_module_path = f"{package_name}.solvers"
        try:
            solvers_module = importlib.import_module(solvers_module_path)
            problem_name = package_name.split('.')[-1]
            
            for _, name, _ in pkgutil.iter_modules(solvers_module.__path__):
                try:
                    module = importlib.import_module(f"{solvers_module_path}.{name}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, SearchAlgorithm) and hasattr(attr, "phase"):
                            phase = getattr(attr, "phase", "unknown")
                            if problem_name not in _solver_registry:
                                _solver_registry[problem_name] = {"exploration": [], "exploitation": []}
                            
                            if phase in _solver_registry[problem_name]:
                                _solver_registry[problem_name][phase].append(attr)

                except ImportError:
                    continue
        except ImportError:
            continue

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
