"""A trivial no-op solver used to disable exploitation phases."""

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


class NoOpSolver(SearchAlgorithm):
    """SearchAlgorithm stub that never updates its population."""

    def __init__(self, problem: ProblemInterface, population_size: int = 1, **kwargs):
        super().__init__(problem, max(1, population_size), **kwargs)

    def initialize(self):
        # Build a population of evaluated initial solutions to be compatible with orchestrator seeding.
        if not self.population:
            self.population = [self.problem.get_initial_solution() for _ in range(self.population_size)]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()

    def step(self):
        # No operation; best solution remains whatever was seeded.
        if not self.population:
            self.initialize()
        self._update_best_solution()
        self.iteration += 1
