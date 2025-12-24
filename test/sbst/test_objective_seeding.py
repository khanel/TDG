from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from Core.problem import ProblemInterface, Solution
from RLOrchestrator.sbst.solvers.explorer import SBSTRandomExplorer


class DummyObjectiveProblem(ProblemInterface):
    """A small fake problem to validate objective-aware reseeding.

    It flips objective tokens after N evaluations and provides a fixed seed for the new objective.
    """

    def __init__(self, *, dimension: int = 6, flip_after: int = 3):
        self._dim = int(dimension)
        self._flip_after = int(flip_after)
        self._evals = 0
        self._active_token: Optional[str] = "A"
        self._rng = np.random.default_rng(0)

    def evaluate(self, solution: Solution) -> float:
        self._evals += 1
        # Flip after a few evals.
        if self._evals >= self._flip_after:
            self._active_token = "B"
        # Fitness doesn't matter for this test.
        return float(self._rng.random())

    def get_initial_solution(self) -> Solution:
        genes = self._rng.integers(0, 10, size=self._dim).tolist()
        sol = Solution({"genes": genes}, self)
        # Let solver evaluate lazily.
        return sol

    def get_problem_info(self) -> Dict[str, Any]:
        return {"dimension": self._dim}

    # --- SBST-like hooks ---
    def get_active_objective_token(self) -> Optional[str]:
        return self._active_token

    def get_population_seeds(self, *, max_seeds: int = 8) -> List[Solution]:
        if self._active_token != "B":
            return []
        # A distinct marker so we can assert it made it into the population.
        genes = [99] * self._dim
        return [Solution({"genes": genes}, self)]


def test_solver_reseeds_population_on_objective_change():
    problem = DummyObjectiveProblem(dimension=6, flip_after=3)
    solver = SBSTRandomExplorer(problem, population_size=8, mutation_rate=0.0, seed=0)

    solver.initialize()
    # Trigger enough evaluations to flip to objective B.
    for _ in range(2):
        solver.step()

    # After reseed, at least one individual should match the seed marker.
    reps = [sol.representation for sol in solver.population]
    assert any(isinstance(r, dict) and r.get("genes") == [99] * 6 for r in reps)
