"""
Shared context/state definitions for the RL orchestrator environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm
from .utils import IntRangeSpec

Phase = Literal["exploration", "exploitation", "termination"]


@dataclass
class StageBinding:
    """Declarative pairing between a stage name and its solver instance."""
    name: Phase
    solver: SearchAlgorithm


@dataclass
class BudgetSpec:
    """Range-based specification of the episode budgets."""
    max_decision_steps: IntRangeSpec
    search_steps_per_decision: IntRangeSpec
    max_search_steps: Optional[int] = None


@dataclass
class OrchestratorContext:
    """
    Source of truth for the orchestrator. Holds the problem instance, registered
    stages, the active RNG, and mutable episode state.
    """
    problem: ProblemInterface
    stages: List[StageBinding]
    budget: BudgetSpec
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    phase_index: int = 0
    decision_count: int = 0
    search_step_count: int = 0
    max_decision_steps: int = 0
    search_steps_per_decision: int = 0
    max_search_steps: Optional[int] = None
    best_solution: Optional[Solution] = None

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("At least one stage must be supplied.")
        for binding in self.stages:
            if binding.name not in ("exploration", "exploitation"):
                raise ValueError(f"Unsupported stage name: {binding.name}")

    def current_stage(self) -> StageBinding:
        return self.stages[self.phase_index]

    def current_solver(self) -> SearchAlgorithm:
        return self.current_stage().solver

    def current_phase(self) -> Phase:
        return self.current_stage().name

    def reset_state(self) -> None:
        self.phase_index = 0
        self.decision_count = 0
        self.search_step_count = 0
        self.best_solution = None


def normalize_range(spec: IntRangeSpec) -> Tuple[int, int]:
    """Convert a scalar or (lo, hi) spec into a well-formed inclusive range."""
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        lo, hi = int(spec[0]), int(spec[1])
    else:
        value = max(1, int(spec))
        return (value, value)
    lo, hi = sorted((lo, hi))
    lo = max(1, lo)
    hi = max(lo, hi)
    return (lo, hi)


def sample_from_range(spec: Tuple[int, int], rng: np.random.Generator) -> int:
    lo, hi = spec
    if lo == hi:
        return lo
    return int(rng.integers(lo, hi + 1))
