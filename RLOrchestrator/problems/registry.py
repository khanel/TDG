"""
Problem registry aligned with the orchestrator context/controller architecture.

Each problem registers a definition that knows how to build the adapter and its
default exploration/exploitation solvers. The registry can then materialize a
`ProblemBundle`, which plugs directly into the `OrchestratorContext`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from Core.problem import ProblemInterface
from Core.search_algorithm import SearchAlgorithm

from ..core.context import Phase, StageBinding

SolverSpec = Union["SolverFactory", Sequence["SolverFactory"]]


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
    solvers: Dict[Phase, SolverSpec] = field(default_factory=dict)
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
            spec = self.solvers.get(phase) if self.solvers else None
            if spec is None:
                continue
            factory = self._select_factory(spec)
            if factory is None:
                continue
            overrides = solver_kwargs.get(phase)
            solver = factory.build(problem, overrides)
            stages.append(StageBinding(name=phase, solver=solver))

        if not stages:
            raise ValueError(f"Problem '{self.name}' does not define any stages.")

        return ProblemBundle(name=self.name, problem=problem, stages=stages)

    @staticmethod
    def _select_factory(spec: SolverSpec) -> Optional["SolverFactory"]:
        if isinstance(spec, (list, tuple)):
            choices = [factory for factory in spec if isinstance(factory, SolverFactory)]
            if not choices:
                return None
            return random.choice(choices)
        return spec if isinstance(spec, SolverFactory) else None


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
    from ..tsp.solvers import (
        # Explorer variants (11 total)
        TSPMapElitesExplorer,
        TSPGAExplorer,
        TSPPSOExplorer,
        TSPGWOExplorer,
        TSPABCExplorer,
        TSPWOAExplorer,
        TSPHHOExplorer,
        TSPMPAExplorer,
        TSPSMAExplorer,
        TSPGSAExplorer,
        TSPDiversityExplorer,
        # Exploiter variants (13 total)
        TSP2OptExploiter,
        TSPGAExploiter,
        TSPPSOExploiter,
        TSPGWOExploiter,
        TSPABCExploiter,
        TSPWOAExploiter,
        TSPHHOExploiter,
        TSPMPAExploiter,
        TSPSMAExploiter,
        TSPGSAExploiter,
        TSPHillClimbingExploiter,
        TSPMemeticExploiter,
        TSPLSHADEExploiter,
    )
    from ..maxcut.adapter import MaxCutAdapter
    from ..maxcut.solvers import (
        # Explorer variants (11 total)
        MaxCutMapElitesExplorer,
        MaxCutGAExplorer,
        MaxCutPSOExplorer,
        MaxCutGWOExplorer,
        MaxCutABCExplorer,
        MaxCutWOAExplorer,
        MaxCutHHOExplorer,
        MaxCutMPAExplorer,
        MaxCutSMAExplorer,
        MaxCutGSAExplorer,
        MaxCutDiversityExplorer,
        # Exploiter variants (13 total)
        MaxCutGreedyExploiter,
        MaxCutGAExploiter,
        MaxCutPSOExploiter,
        MaxCutGWOExploiter,
        MaxCutABCExploiter,
        MaxCutWOAExploiter,
        MaxCutHHOExploiter,
        MaxCutMPAExploiter,
        MaxCutSMAExploiter,
        MaxCutGSAExploiter,
        MaxCutHillClimbingExploiter,
        MaxCutMemeticExploiter,
        MaxCutLSHADEExploiter,
    )
    from ..knapsack.adapter import KnapsackAdapter
    from ..knapsack.solvers import (
        # Explorer variants (11 total)
        KnapsackMapElitesExplorer,
        KnapsackGAExplorer,
        KnapsackPSOExplorer,
        KnapsackGWOExplorer,
        KnapsackABCExplorer,
        KnapsackWOAExplorer,
        KnapsackHHOExplorer,
        KnapsackMPAExplorer,
        KnapsackSMAExplorer,
        KnapsackGSAExplorer,
        KnapsackDiversityExplorer,
        # Exploiter variants (13 total)
        KnapsackGreedyExploiter,
        KnapsackGAExploiter,
        KnapsackPSOExploiter,
        KnapsackGWOExploiter,
        KnapsackABCExploiter,
        KnapsackWOAExploiter,
        KnapsackHHOExploiter,
        KnapsackMPAExploiter,
        KnapsackSMAExploiter,
        KnapsackGSAExploiter,
        KnapsackHillClimbingExploiter,
        KnapsackMemeticExploiter,
        KnapsackLSHADEExploiter,
    )
    from ..nkl.adapter import NKLAdapter
    from ..nkl.solvers import (
        # Explorer variants (properly tuned for diversity/global search)
        NKLMapElitesExplorer,
        NKLGWOExplorer,
        NKLPSOExplorer,
        NKLGAExplorer,
        NKLABCExplorer,
        NKLWOAExplorer,
        NKLHHOExplorer,
        NKLMPAExplorer,
        NKLSMAExplorer,
        NKLGSAExplorer,
        NKLDiversityExplorer,
        # Exploiter variants (properly tuned for convergence/local refinement)
        NKLBinaryPSOExploiter,
        NKLGWOExploiter,
        NKLPSOExploiter,
        NKLGAExploiter,
        NKLLSHADEExploiter,
        NKLWOAExploiter,
        NKLHHOExploiter,
        NKLMPAExploiter,
        NKLSMAExploiter,
        NKLGSAExploiter,
        NKLHillClimbingExploiter,
        NKLMemeticExploiter,
        NKLABCExploiter,
    )

    from ..sbst.adapter import SBSTAdapter
    from ..sbst.solvers import SBSTRandomExplorer, SBSTRandomExploiter

    # TSP: Full 11x13 explorer/exploiter pool for solver-agnostic training
    tsp_explorers = [
        SolverFactory(TSPMapElitesExplorer, {"population_size": 64, "n_bins": 10, "mutation_rate": 0.3}),
        SolverFactory(TSPGAExplorer, {"population_size": 64, "mutation_rate": 0.4, "random_injection": 0.1}),
        SolverFactory(TSPPSOExplorer, {"population_size": 56, "cognitive": 0.7, "social": 0.3, "random_moves": 3}),
        SolverFactory(TSPGWOExplorer, {"population_size": 48, "a_decay": 0.5, "random_weight": 0.4}),
        SolverFactory(TSPABCExplorer, {"population_size": 72, "limit": 5, "scout_rate": 0.2}),
        SolverFactory(TSPWOAExplorer, {"population_size": 48, "b": 1.0, "a_decay": 0.5}),
        SolverFactory(TSPHHOExplorer, {"population_size": 56, "energy_decay": 0.5}),
        SolverFactory(TSPMPAExplorer, {"population_size": 60, "FADs": 0.3}),
        SolverFactory(TSPSMAExplorer, {"population_size": 64, "z": 0.03}),
        SolverFactory(TSPGSAExplorer, {"population_size": 56, "G0": 50.0, "alpha": 10.0}),
        SolverFactory(TSPDiversityExplorer, {"population_size": 48, "sigma": 0.3, "alpha": 1.0}),
    ]

    tsp_exploiters = [
        SolverFactory(TSP2OptExploiter, {"population_size": 48, "elite_ratio": 0.4, "local_search_iter": 30}),
        SolverFactory(TSPGAExploiter, {"population_size": 56, "mutation_rate": 0.1, "elite_ratio": 0.3}),
        SolverFactory(TSPPSOExploiter, {"population_size": 48, "cognitive": 0.3, "social": 0.7, "local_search_iter": 10}),
        SolverFactory(TSPGWOExploiter, {"population_size": 40, "a_decay": 2.0}),
        SolverFactory(TSPABCExploiter, {"population_size": 60, "limit": 20, "local_search_iter": 15}),
        SolverFactory(TSPWOAExploiter, {"population_size": 36, "b": 1.0}),
        SolverFactory(TSPHHOExploiter, {"population_size": 40}),
        SolverFactory(TSPMPAExploiter, {"population_size": 48}),
        SolverFactory(TSPSMAExploiter, {"population_size": 56}),
        SolverFactory(TSPGSAExploiter, {"population_size": 48, "G0": 100.0}),
        SolverFactory(TSPHillClimbingExploiter, {"population_size": 32, "local_search_iter": 50}),
        SolverFactory(TSPMemeticExploiter, {"population_size": 40, "mutation_rate": 0.1, "local_search_iter": 20}),
        SolverFactory(TSPLSHADEExploiter, {"population_size": 64, "H": 5}),
    ]

    register_problem(
        ProblemDefinition(
            name="tsp",
            adapter_cls=TSPAdapter,
            default_adapter_kwargs={"num_cities": 20, "grid_size": 100.0},
            solvers={
                "exploration": tsp_explorers,
                "exploitation": tsp_exploiters,
            },
            metadata={
                "description": "TSP with full 11x13 properly-tuned explorer/exploiter pool.",
                "explorer_count": 11,
                "exploiter_count": 13,
                "total_pairings": 11 * 13,  # 143 unique solver combinations
            },
        )
    )

    # MaxCut: Full 11x13 explorer/exploiter pool for solver-agnostic training
    maxcut_explorers = [
        SolverFactory(MaxCutMapElitesExplorer, {"population_size": 64, "n_bins": 10, "mutation_rate": 0.15}),
        SolverFactory(MaxCutGAExplorer, {"population_size": 64, "mutation_rate": 0.15, "crossover_rate": 0.7, "random_injection_rate": 0.1}),
        SolverFactory(MaxCutPSOExplorer, {"population_size": 56, "w": 0.9, "c1": 2.5, "c2": 0.5}),
        SolverFactory(MaxCutGWOExplorer, {"population_size": 48, "a_initial": 3.0, "a_final": 1.0, "random_wolf_prob": 0.3}),
        SolverFactory(MaxCutABCExplorer, {"population_size": 72, "limit_factor": 0.5, "perturbation_scale": 0.8}),
        SolverFactory(MaxCutWOAExplorer, {"population_size": 48, "a_initial": 3.0, "encircle_prob": 0.3}),
        SolverFactory(MaxCutHHOExplorer, {"population_size": 56, "exploration_bias": 0.7, "random_hawk_prob": 0.5}),
        SolverFactory(MaxCutMPAExplorer, {"population_size": 60, "fad_probability": 0.4, "brownian_scale": 1.5}),
        SolverFactory(MaxCutSMAExplorer, {"population_size": 64, "random_position_prob": 0.4, "mutation_rate": 0.15}),
        SolverFactory(MaxCutGSAExplorer, {"population_size": 56, "G0": 200.0, "alpha": 10.0}),
        SolverFactory(MaxCutDiversityExplorer, {"population_size": 48, "mutation_rate": 0.2, "random_injection_rate": 0.25}),
    ]

    maxcut_exploiters = [
        SolverFactory(MaxCutGreedyExploiter, {"population_size": 32, "n_flips": 10}),
        SolverFactory(MaxCutGAExploiter, {"population_size": 56, "mutation_rate": 0.02, "elitism_rate": 0.1}),
        SolverFactory(MaxCutPSOExploiter, {"population_size": 48, "w": 0.4, "c1": 1.0, "c2": 2.5}),
        SolverFactory(MaxCutGWOExploiter, {"population_size": 40, "a_initial": 2.0, "a_final": 0.0}),
        SolverFactory(MaxCutABCExploiter, {"population_size": 60, "limit_factor": 3.0, "perturbation_scale": 0.2}),
        SolverFactory(MaxCutWOAExploiter, {"population_size": 36, "a_initial": 2.0, "encircle_prob": 0.7, "spiral_b": 0.5}),
        SolverFactory(MaxCutHHOExploiter, {"population_size": 40, "exploitation_bias": 0.8}),
        SolverFactory(MaxCutMPAExploiter, {"population_size": 48, "fad_probability": 0.05, "levy_scale": 0.5}),
        SolverFactory(MaxCutSMAExploiter, {"population_size": 56, "random_position_prob": 0.03}),
        SolverFactory(MaxCutGSAExploiter, {"population_size": 48, "G0": 100.0, "alpha": 30.0}),
        SolverFactory(MaxCutHillClimbingExploiter, {"population_size": 24, "n_neighbors": 10}),
        SolverFactory(MaxCutMemeticExploiter, {"population_size": 40, "local_search_prob": 0.8, "local_search_steps": 10}),
        SolverFactory(MaxCutLSHADEExploiter, {"population_size": 64, "p_best_rate": 0.05}),
    ]

    register_problem(
        ProblemDefinition(
            name="maxcut",
            adapter_cls=MaxCutAdapter,
            default_adapter_kwargs={"n_nodes": 64, "edge_probability": 0.5},
            solvers={
                "exploration": maxcut_explorers,
                "exploitation": maxcut_exploiters,
            },
            metadata={
                "description": "MaxCut with full 11x13 properly-tuned explorer/exploiter pool.",
                "explorer_count": 11,
                "exploiter_count": 13,
                "total_pairings": 11 * 13,  # 143 unique solver combinations
            },
        )
    )

    # Knapsack: Full 11x13 explorer/exploiter pool for solver-agnostic training
    knapsack_explorers = [
        SolverFactory(KnapsackMapElitesExplorer, {"population_size": 64, "n_bins": 10, "mutation_rate": 0.15}),
        SolverFactory(KnapsackGAExplorer, {"population_size": 64, "mutation_rate": 0.15, "crossover_rate": 0.7, "random_injection_rate": 0.1}),
        SolverFactory(KnapsackPSOExplorer, {"population_size": 56, "w": 0.9, "c1": 2.5, "c2": 0.5}),
        SolverFactory(KnapsackGWOExplorer, {"population_size": 48, "a_initial": 3.0, "a_final": 1.0, "random_wolf_prob": 0.3}),
        SolverFactory(KnapsackABCExplorer, {"population_size": 72, "limit_factor": 0.5, "perturbation_scale": 0.8}),
        SolverFactory(KnapsackWOAExplorer, {"population_size": 48, "a_initial": 3.0, "encircle_prob": 0.3}),
        SolverFactory(KnapsackHHOExplorer, {"population_size": 56, "exploration_bias": 0.7, "random_hawk_prob": 0.5}),
        SolverFactory(KnapsackMPAExplorer, {"population_size": 60, "fad_probability": 0.4, "brownian_scale": 1.5}),
        SolverFactory(KnapsackSMAExplorer, {"population_size": 64, "random_position_prob": 0.4, "mutation_rate": 0.15}),
        SolverFactory(KnapsackGSAExplorer, {"population_size": 56, "G0": 200.0, "alpha": 10.0}),
        SolverFactory(KnapsackDiversityExplorer, {"population_size": 48, "mutation_rate": 0.2, "random_injection_rate": 0.25}),
    ]

    knapsack_exploiters = [
        SolverFactory(KnapsackGreedyExploiter, {"population_size": 32, "n_flips": 10}),
        SolverFactory(KnapsackGAExploiter, {"population_size": 56, "mutation_rate": 0.02, "elitism_rate": 0.1}),
        SolverFactory(KnapsackPSOExploiter, {"population_size": 48, "w": 0.4, "c1": 1.0, "c2": 2.5}),
        SolverFactory(KnapsackGWOExploiter, {"population_size": 40, "a_initial": 2.0, "a_final": 0.0}),
        SolverFactory(KnapsackABCExploiter, {"population_size": 60, "limit_factor": 3.0, "perturbation_scale": 0.2}),
        SolverFactory(KnapsackWOAExploiter, {"population_size": 36, "a_initial": 2.0, "encircle_prob": 0.7, "spiral_b": 0.5}),
        SolverFactory(KnapsackHHOExploiter, {"population_size": 40, "exploitation_bias": 0.8}),
        SolverFactory(KnapsackMPAExploiter, {"population_size": 48, "fad_probability": 0.05, "levy_scale": 0.5}),
        SolverFactory(KnapsackSMAExploiter, {"population_size": 56, "random_position_prob": 0.03}),
        SolverFactory(KnapsackGSAExploiter, {"population_size": 48, "G0": 100.0, "alpha": 30.0}),
        SolverFactory(KnapsackHillClimbingExploiter, {"population_size": 24, "n_neighbors": 10}),
        SolverFactory(KnapsackMemeticExploiter, {"population_size": 40, "local_search_prob": 0.8, "local_search_steps": 10}),
        SolverFactory(KnapsackLSHADEExploiter, {"population_size": 64, "p_best_rate": 0.05}),
    ]

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
                "exploration": knapsack_explorers,
                "exploitation": knapsack_exploiters,
            },
            metadata={
                "description": "Binary knapsack with full 11x13 properly-tuned explorer/exploiter pool.",
                "explorer_count": 11,
                "exploiter_count": 13,
                "total_pairings": 11 * 13,  # 143 unique solver combinations
            },
        )
    )

    # NKL: Full 11x13 explorer/exploiter pool for solver-agnostic training
    # Each solver is properly tuned for its role (exploration vs exploitation)
    nkl_explorers = [
        # Quality-Diversity (QD) - maintains diverse archive
        SolverFactory(NKLMapElitesExplorer, {"population_size": 64, "n_bins": 10, "mutation_rate": 0.1}),
        # Population-based metaheuristics tuned for exploration
        SolverFactory(NKLGWOExplorer, {"population_size": 48, "a_initial": 3.0, "a_final": 1.0, "mutation_rate": 0.15}),
        SolverFactory(NKLPSOExplorer, {"population_size": 56, "omega": 0.9, "c1": 2.5, "c2": 0.5}),
        SolverFactory(NKLGAExplorer, {"population_size": 64, "mutation_rate": 0.15, "random_immigrant_rate": 0.1}),
        SolverFactory(NKLABCExplorer, {"population_size": 72, "limit_factor": 0.5, "perturbation_scale": 0.8}),
        SolverFactory(NKLWOAExplorer, {"population_size": 48, "a_initial": 3.0, "encircle_prob": 0.3}),
        SolverFactory(NKLHHOExplorer, {"population_size": 56, "exploration_bias": 0.7, "random_hawk_prob": 0.5}),
        SolverFactory(NKLMPAExplorer, {"population_size": 60, "fad_probability": 0.4, "brownian_scale": 1.5}),
        SolverFactory(NKLSMAExplorer, {"population_size": 64, "random_position_prob": 0.4, "mutation_rate": 0.15}),
        SolverFactory(NKLGSAExplorer, {"population_size": 56, "G0": 200.0, "alpha": 10.0, "mutation_rate": 0.1}),
        SolverFactory(NKLDiversityExplorer, {"population_size": 48, "mutation_rate": 0.2, "random_injection_rate": 0.25}),
    ]

    nkl_exploiters = [
        # Binary-specific local search
        SolverFactory(NKLBinaryPSOExploiter, {"population_size": 32}),
        SolverFactory(NKLHillClimbingExploiter, {"population_size": 24, "n_neighbors": 10}),
        SolverFactory(NKLMemeticExploiter, {"population_size": 40, "local_search_prob": 0.8, "local_search_steps": 10}),
        # Population-based metaheuristics tuned for exploitation
        SolverFactory(NKLGWOExploiter, {"population_size": 40, "a_initial": 2.0, "a_final": 0.0, "mutation_rate": 0.02}),
        SolverFactory(NKLPSOExploiter, {"population_size": 48, "omega": 0.4, "c1": 1.0, "c2": 2.5}),
        SolverFactory(NKLGAExploiter, {"population_size": 56, "mutation_rate": 0.02, "elitism_rate": 0.1}),
        SolverFactory(NKLLSHADEExploiter, {"population_size": 64, "p_best_rate": 0.05}),
        SolverFactory(NKLWOAExploiter, {"population_size": 36, "a_initial": 2.0, "encircle_prob": 0.7, "spiral_b": 0.5}),
        SolverFactory(NKLHHOExploiter, {"population_size": 40, "exploitation_bias": 0.8, "levy_steps": 3}),
        SolverFactory(NKLMPAExploiter, {"population_size": 48, "fad_probability": 0.05, "levy_scale": 0.5}),
        SolverFactory(NKLSMAExploiter, {"population_size": 56, "random_position_prob": 0.03, "mutation_rate": 0.02}),
        SolverFactory(NKLGSAExploiter, {"population_size": 48, "G0": 100.0, "alpha": 30.0, "mutation_rate": 0.02}),
        SolverFactory(NKLABCExploiter, {"population_size": 60, "limit_factor": 3.0, "perturbation_scale": 0.2}),
    ]

    register_problem(
        ProblemDefinition(
            name="nkl",
            adapter_cls=NKLAdapter,
            default_adapter_kwargs={"n_items": 100, "k_interactions": 5},
            solvers={
                "exploration": nkl_explorers,
                "exploitation": nkl_exploiters,
            },
            metadata={
                "description": "NK-Landscape with full 11x13 properly-tuned explorer/exploiter pool.",
                "explorer_count": 11,
                "exploiter_count": 13,
                "total_pairings": 11 * 13,  # 143 unique solver combinations
            },
        )
    )

    # SBST TDG: minimal solver pair so SBST can be exercised end-to-end via the registry.
    # The adapter evaluation is backed by the Java pipeline (JUnit + Maven/Gradle + JaCoCo).
    sbst_explorers = [
        SolverFactory(SBSTRandomExplorer, {"population_size": 48, "mutation_rate": 0.35}),
    ]
    sbst_exploiters = [
        SolverFactory(SBSTRandomExploiter, {"population_size": 32, "mutation_rate": 0.10}),
    ]

    register_problem(
        ProblemDefinition(
            name="sbst",
            adapter_cls=SBSTAdapter,
            default_adapter_kwargs={"dimension": 24},
            solvers={
                "exploration": sbst_explorers,
                "exploitation": sbst_exploiters,
            },
            metadata={
                "description": "SBST TDG (Java) — end-to-end JUnit generation + Maven/Gradle execution + JaCoCo branch fitness.",
                "explorer_count": 1,
                "exploiter_count": 1,
                "total_pairings": 1,
            },
        )
    )
