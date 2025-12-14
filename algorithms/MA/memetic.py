"""
Memetic Algorithm (MA) implementation.

Combines a genetic algorithm with localized improvements (memes) to refine
offspring after variation. Uses `ProblemInterface.sample_neighbors` when
available, otherwise falls back to generic perturbations.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


class MemeticAlgorithm(SearchAlgorithm):
    """Hybrid GA + local search (memes)."""

    phase = "exploitation"

    def __init__(
        self,
        problem,
        population_size: int,
        *,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        local_search_probability: float = 0.3,
        local_search_steps: int = 3,
        seed: Optional[int] = None,
    ):
        super().__init__(problem, population_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.local_search_probability = local_search_probability
        self.local_search_steps = local_search_steps
        self.rng = np.random.default_rng(seed)

    def step(self):
        if not self.population:
            self.initialize()
        parents = [self._tournament_select() for _ in range(self.population_size)]
        offspring: List[Solution] = []
        for i in range(0, self.population_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % self.population_size]
            children = self._crossover(p1, p2)
            offspring.extend(children)
        offspring = [self._mutate(child) for child in offspring]
        offspring = [self._local_search(child) for child in offspring]
        combined = self.population + offspring
        combined.sort(key=lambda sol: sol.fitness if sol.fitness is not None else sol.evaluate())
        self.population = combined[: self.population_size]
        self._update_best_solution()
        self.iteration += 1

    # --- Genetic operators -------------------------------------------------
    def _tournament_select(self) -> Solution:
        contenders = self.rng.choice(self.population, size=self.tournament_size, replace=False)
        best = min(contenders, key=lambda sol: sol.fitness if sol.fitness is not None else sol.evaluate())
        return best.copy(preserve_id=False)

    def _crossover(self, parent1: Solution, parent2: Solution):
        if self.rng.random() > self.crossover_rate:
            return [parent1.copy(preserve_id=False), parent2.copy(preserve_id=False)]
        rep1 = np.asarray(parent1.representation)
        rep2 = np.asarray(parent2.representation)
        if rep1.shape != rep2.shape:
            return [parent1.copy(preserve_id=False), parent2.copy(preserve_id=False)]
        mask = self.rng.random(rep1.shape) < 0.5
        child1 = np.where(mask, rep1, rep2)
        child2 = np.where(mask, rep2, rep1)
        return [Solution(child1.copy(), self.problem), Solution(child2.copy(), self.problem)]

    def _mutate(self, solution: Solution) -> Solution:
        rep = np.asarray(solution.representation).copy()
        if self.rng.random() > self.mutation_rate:
            solution.evaluate()
            return solution
        if rep.ndim == 1 and rep.dtype.kind in {"i", "u"} and len(np.unique(rep)) == len(rep):
            i, j = self.rng.choice(len(rep), size=2, replace=False)
            rep[i], rep[j] = rep[j], rep[i]
        else:
            noise = self.rng.normal(0, 1, size=rep.shape)
            rep = rep + 0.1 * noise
        mutated = Solution(rep.tolist() if isinstance(solution.representation, list) else rep, self.problem)
        mutated.evaluate()
        return mutated

    # --- Local search ------------------------------------------------------
    def _local_search(self, solution: Solution) -> Solution:
        solution.evaluate()
        if self.rng.random() > self.local_search_probability:
            return solution
        current = solution
        for _ in range(self.local_search_steps):
            neighbor = self._sample_neighbor(current)
            if neighbor.fitness < current.fitness:
                current = neighbor
        return current

    def _sample_neighbor(self, solution: Solution) -> Solution:
        if hasattr(self.problem, "sample_neighbors"):
            neighbors = self.problem.sample_neighbors(solution, 1)
            if neighbors:
                neighbor = neighbors[0]
                neighbor.evaluate()
                return neighbor
        rep = np.asarray(solution.representation).copy()
        noise = self.rng.normal(0, 0.05, size=rep.shape)
        neighbor = Solution(rep + noise, self.problem)
        neighbor.evaluate()
        return neighbor
