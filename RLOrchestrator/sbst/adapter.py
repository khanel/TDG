"""SBST TDG adapter (placeholder).

This adapter is a *scaffold* that fits the existing `ProblemInterface` contract so the
RL orchestrator can treat SBST TDG like other problems (TSP/MaxCut/Knapsack/NKL).

Real SBST evaluation (JUnit generation + JaCoCo branch coverage parsing) will
replace the placeholder fitness function in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from Core.problem import ProblemInterface, Solution


@dataclass(frozen=True)
class SBSTConfig:
    """Minimal configuration for the SBST problem scaffold."""

    dimension: int = 24
    seed: Optional[int] = 42


class SBSTAdapter(ProblemInterface):
    """Temporary SBST adapter.

    Representation (current scaffold):
    - `solution.representation` is a dict with a `genes: list[int]` field.

    Fitness (current scaffold):
    - Minimizes a smooth surrogate objective to keep solvers functional.
    - Later replaced with: `1 - branch_coverage(target_class)`.
    """

    def __init__(self, *, dimension: int = 24, seed: Optional[int] = 42):
        self._cfg = SBSTConfig(dimension=max(1, int(dimension)), seed=seed)
        self._rng = np.random.default_rng(seed)
        self._bounds = {"lower_bound": 0.0, "upper_bound": 1.0}

    def evaluate(self, solution: Solution) -> float:
        rep = solution.representation
        genes: List[int]

        if isinstance(rep, dict) and "genes" in rep:
            genes = list(rep["genes"])
        elif isinstance(rep, list):
            genes = list(rep)
        else:
            genes = [0] * self._cfg.dimension

        # Surrogate objective in [0, 1]: mean squared distance from 0.5.
        arr = np.asarray(genes, dtype=float)
        if arr.size == 0:
            fitness = 1.0
        else:
            scaled = (arr % 10) / 9.0  # map ints to [0,1]
            fitness = float(np.clip(np.mean((scaled - 0.5) ** 2) * 4.0, 0.0, 1.0))

        solution.fitness = fitness
        return fitness

    def get_initial_solution(self) -> Solution:
        genes = self._rng.integers(0, 100, size=self._cfg.dimension).tolist()
        sol = Solution({"genes": genes}, self)
        sol.evaluate()
        return sol

    def get_initial_population(self, population_size: int) -> List[Solution]:
        return [self.get_initial_solution() for _ in range(max(1, int(population_size)))]

    def get_problem_info(self) -> Dict[str, Any]:
        return {
            "dimension": int(self._cfg.dimension),
            "problem_type": "mixed",  # placeholder; SBST candidates will be structured objects
            "objective": "minimize_surrogate_until_jacoco_wired",
        }

    def get_bounds(self) -> Dict[str, float]:
        return dict(self._bounds)
