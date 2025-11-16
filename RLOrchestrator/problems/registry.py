"""
Problem registry aligned with the orchestrator context/controller architecture.

Each problem registers a definition that knows how to build the adapter and its
default exploration/exploitation solvers. The registry can then materialize a
`ProblemBundle`, which plugs directly into the `OrchestratorContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from Core.problem import ProblemInterface
from Core.search_algorithm import SearchAlgorithm

from ..core.context import Phase, StageBinding


@dataclass
class SolverFactory:
    """Describes how to instantiate a solver for a stage."""
    cls: Type[SearchAlgorithm]
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(
        self,
        problem: ProblemInterface,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> SearchAlgorithm:
        params: Dict[str, Any] = dict(self.default_kwargs)
        if overrides:
            params.update(overrides)
        return self.cls(problem, **params)


@dataclass
class ProblemDefinition:
    """Declarative description of a problem + its default solver pair."""
    name: str
    adapter_cls: Type[ProblemInterface]
    default_adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
    solvers: Dict[Phase, SolverFactory] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def instantiate(
        self,
        *,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        solver_kwargs: Optional[Dict[Phase, Dict[str, Any]]] = None,
    ) -> "ProblemBundle":
        """Materialize the adapter and its stage solvers."""
        adapter_params = dict(self.default_adapter_kwargs)
        if adapter_kwargs:
            adapter_params.update(adapter_kwargs)
        problem = self.adapter_cls(**adapter_params)

        stages: List[StageBinding] = []
        solver_kwargs = solver_kwargs or {}

        for phase in ("exploration", "exploitation"):
            factory = self.solvers.get(phase) if self.solvers else None
            if factory is None:
                continue
            overrides = solver_kwargs.get(phase)
            solver = factory.build(problem, overrides)
            stages.append(StageBinding(name=phase, solver=solver))

        if not stages:
            raise ValueError(f"Problem '{self.name}' does not define any stages.")

        return ProblemBundle(name=self.name, problem=problem, stages=stages)


@dataclass
class ProblemBundle:
    """Concrete adapter + solver bindings ready for a context."""
    name: str
    problem: ProblemInterface
    stages: List[StageBinding]


_problem_definitions: Dict[str, ProblemDefinition] = {}
_BUILTINS_REGISTERED = False


def register_problem(definition: ProblemDefinition) -> None:
    """Register (or override) a problem definition."""
    _problem_definitions[definition.name] = definition


def get_problem_definition(name: str) -> Optional[ProblemDefinition]:
    _ensure_builtin_definitions()
    return _problem_definitions.get(name)


def list_problem_definitions() -> Dict[str, ProblemDefinition]:
    _ensure_builtin_definitions()
    return dict(_problem_definitions)


def instantiate_problem(
    name: str,
    *,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
    solver_kwargs: Optional[Dict[Phase, Dict[str, Any]]] = None,
) -> ProblemBundle:
    definition = get_problem_definition(name)
    if definition is None:
        raise KeyError(f"Problem '{name}' is not registered.")
    return definition.instantiate(adapter_kwargs=adapter_kwargs, solver_kwargs=solver_kwargs)


# --- Backward-compatibility helpers ---------------------------------------

def get_problem_adapter(name: str) -> Optional[Type[ProblemInterface]]:
    """Legacy helper returning only the adapter class."""
    definition = get_problem_definition(name)
    return definition.adapter_cls if definition else None


def get_problem_registry() -> Dict[str, Type[ProblemInterface]]:
    """Legacy helper returning adapter classes keyed by name."""
    _ensure_builtin_definitions()
    return {name: definition.adapter_cls for name, definition in _problem_definitions.items()}


# --- Built-in definitions -------------------------------------------------

def _ensure_builtin_definitions():
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _register_builtin_definitions()
    _BUILTINS_REGISTERED = True


def _register_builtin_definitions():
    from ..tsp.adapter import TSPAdapter
    from ..tsp.solvers.map_elites import TSPMapElites
    from ..tsp.solvers.pso import TSPParticleSwarm
    from ..maxcut.adapter import MaxCutAdapter
    from ..maxcut.solvers.explorer import MaxCutRandomExplorer
    from ..maxcut.solvers.exploiter import MaxCutBitFlipExploiter
    from ..knapsack.adapter import KnapsackAdapter
    from ..knapsack.solvers.explorer import KnapsackRandomExplorer
    from ..knapsack.solvers.exploiter import KnapsackBitFlipExploiter
    from ..nkl.adapter import NKLAdapter
    from ..nkl.solvers.explorer import NKLRandomExplorer
    from ..nkl.solvers.exploiter import NKLBitFlipExploiter

    register_problem(
        ProblemDefinition(
            name="tsp",
            adapter_cls=TSPAdapter,
            default_adapter_kwargs={"num_cities": 20, "grid_size": 100.0},
            solvers={
                "exploration": SolverFactory(TSPMapElites, {"population_size": 64}),
                "exploitation": SolverFactory(TSPParticleSwarm, {"population_size": 32}),
            },
            metadata={
                "description": "Traveling Salesperson Problem baseline configuration.",
                "observation_features": 6,
            },
        )
    )

    register_problem(
        ProblemDefinition(
            name="maxcut",
            adapter_cls=MaxCutAdapter,
            default_adapter_kwargs={"n_nodes": 64, "edge_probability": 0.5},
            solvers={
                "exploration": SolverFactory(MaxCutRandomExplorer, {"population_size": 64}),
                "exploitation": SolverFactory(MaxCutBitFlipExploiter, {"population_size": 16}),
            },
            metadata={"description": "Erdos-Renyi Max-Cut baseline."},
        )
    )

    register_problem(
        ProblemDefinition(
            name="knapsack",
            adapter_cls=KnapsackAdapter,
            default_adapter_kwargs={
                "n_items": 50,
                "value_range": (1.0, 100.0),
                "weight_range": (1.0, 50.0),
                "capacity_ratio": 0.5,
            },
            solvers={
                "exploration": SolverFactory(KnapsackRandomExplorer, {"population_size": 64}),
                "exploitation": SolverFactory(KnapsackBitFlipExploiter, {"population_size": 16}),
            },
            metadata={"description": "Binary knapsack baseline with random/value-weight initialization."},
        )
    )

    register_problem(
        ProblemDefinition(
            name="nkl",
            adapter_cls=NKLAdapter,
            default_adapter_kwargs={"n_items": 100, "k_interactions": 5},
            solvers={
                "exploration": SolverFactory(NKLRandomExplorer, {"population_size": 64}),
                "exploitation": SolverFactory(NKLBitFlipExploiter, {"population_size": 16}),
            },
            metadata={"description": "NK-Landscape baseline with vectorized solvers."},
        )
    )
