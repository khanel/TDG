"""Solver registry bootstrap with lazy default registrations."""

from importlib import import_module
from typing import Dict

from .registry import (
    get_solver_registry as _get_solver_registry,
    register_exploration_solver,
    register_exploitation_solver,
)

_DEFAULT_EXPLORATION: Dict[str, str] = {
    "sa": "SA.SA:SimulatedAnnealing",
    "tsp_map_elites": "RLOrchestrator.tsp.solvers.map_elites:TSPMapElites",
    "noop": "RLOrchestrator.solvers.noop:NoOpSolver",
    "maxcut_random": "RLOrchestrator.maxcut.solvers.explorer:MaxCutRandomExplorer",
    "knapsack_random": "RLOrchestrator.knapsack.solvers.explorer:KnapsackRandomExplorer",
}

_DEFAULT_EXPLOITATION: Dict[str, str] = {
    "sa": "SA.SA:SimulatedAnnealing",
    "noop": "RLOrchestrator.solvers.noop:NoOpSolver",
    "tsp_pso": "RLOrchestrator.tsp.solvers.pso:TSPParticleSwarm",
    "pso": "PSO.PSO:ParticleSwarmOptimization",
    "maxcut_local": "RLOrchestrator.maxcut.solvers.local_search:MaxCutLocalSearch",
    "knapsack_local": "RLOrchestrator.knapsack.solvers.local_search:KnapsackLocalSearch",
}


def _resolve(path: str):
    module_name, attr = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr)


_defaults_registered = False


def _ensure_defaults() -> None:
    global _defaults_registered
    if _defaults_registered:
        return
    for name, target in _DEFAULT_EXPLORATION.items():
        register_exploration_solver(name, _resolve(target))
    for name, target in _DEFAULT_EXPLOITATION.items():
        register_exploitation_solver(name, _resolve(target))
    _defaults_registered = True


def get_solver_registry():
    _ensure_defaults()
    return _get_solver_registry()


__all__ = [
    'get_solver_registry',
    'register_exploration_solver',
    'register_exploitation_solver',
]
