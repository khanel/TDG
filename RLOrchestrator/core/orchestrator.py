"""
Phased orchestrator compatible with root Core APIs.
Wraps two Core.search_algorithm.SearchAlgorithm instances: exploration and exploitation.
"""

from typing import Optional, List, Literal
from Core.problem import Solution, ProblemInterface
from Core.search_algorithm import SearchAlgorithm


Phase = Literal["exploration", "exploitation"]


class Orchestrator:
    """Orchestrates between exploration and exploitation solvers."""

    def __init__(
        self,
        problem: ProblemInterface,
        exploration_solver: SearchAlgorithm,
        exploitation_solver: SearchAlgorithm,
        *,
        start_phase: Phase = "exploration",
    ):
        self.problem = problem
        self.exploration_solver = exploration_solver
        self.exploitation_solver = exploitation_solver
        if start_phase not in ("exploration", "exploitation"):
            raise ValueError(f"Invalid start phase: {start_phase}")
        self.phase: Phase = start_phase
        self._best_solution: Optional[Solution] = None

    def get_phase(self) -> Phase:
        """Return current phase."""
        return self.phase

    def step(self, n_steps: int = 1) -> None:
        """Run current phase solver for n_steps (looping if needed)."""
        algo = self.exploration_solver if self.phase == "exploration" else self.exploitation_solver
        # Core SearchAlgorithm.step() has no n_steps parameter; loop if n_steps > 1
        for _ in range(max(1, int(n_steps))):
            algo.step()
        self._update_best()

    def switch_to_exploitation(self, seeds: Optional[List[Solution]] = None) -> None:
        """Switch to exploitation phase, optionally seeding with solutions."""
        if self.phase == "exploitation":
            return
        if seeds is None:
            pop = self.exploration_solver.get_population()
            seeds = [s for s in pop if s is not None]
        if hasattr(self.exploitation_solver, "ingest_seeds"):
            self.exploitation_solver.ingest_seeds(seeds)
        else:
            # Seed exploitation by replacing its population; then refresh its best
            self.exploitation_solver.population = [s.copy() for s in seeds]
            # Ensure fitness is evaluated and best is updated if helper exists
            if hasattr(self.exploitation_solver, "_update_best_solution"):
                self.exploitation_solver._update_best_solution()
        self.phase = "exploitation"
        self._update_best()

    def get_current_solver(self):
        """Return the active solver."""
        return self.exploration_solver if self.phase == "exploration" else self.exploitation_solver

    def get_best_solution(self) -> Optional[Solution]:
        """Return the best solution across phases."""
        return self._best_solution

    def _update_best(self) -> None:
        """Update the global best solution."""
        exp_best = self.exploration_solver.get_best()
        exp_fit = exp_best.fitness if exp_best else float("inf")
        exp_sol = self.exploitation_solver.get_best()
        exp_sol_fit = exp_sol.fitness if exp_sol else float("inf")
        if exp_fit < exp_sol_fit:
            self._best_solution = exp_best
        else:
            self._best_solution = exp_sol
