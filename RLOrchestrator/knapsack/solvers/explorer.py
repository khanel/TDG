"""Exploration solver for Knapsack using stochastic bit flips."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


class KnapsackRandomExplorer(SearchAlgorithm):
    """Maintains a diverse population via random bit flips and repair."""
    phase = 'exploration'

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 64,
        *,
        flip_probability: float = 0.15,
        elite_fraction: float = 0.25,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackRandomExplorer expects a KnapsackAdapter exposing `knapsack_problem`.")
        super().__init__(problem, population_size)
        self.flip_probability = float(np.clip(flip_probability, 0.01, 0.9))
        self.elite_fraction = float(np.clip(elite_fraction, 0.05, 0.8))
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)

    def step(self):
        self._ensure_evaluated(self.population)
        elite_count = max(1, int(self.elite_fraction * len(self.population)))
        sorted_pop = sorted(self.population, key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        elites = [sol.copy(preserve_id=False) for sol in sorted_pop[:elite_count]]

        offspring: List[Solution] = []

        for parent in self.population:
            mask = np.asarray(parent.representation, dtype=int)
            flips = self.rng.random(mask.shape[0]) < self.flip_probability
            if not np.any(flips):
                flips[self.rng.integers(mask.shape[0])] = True
            child_mask = mask.copy()
            child_mask[flips] = 1 - child_mask[flips]
            child_mask = np.asarray(self.problem.repair_mask(child_mask.tolist()), dtype=int)
            child = Solution(child_mask.tolist(), self.problem)
            child.evaluate()
            offspring.append(child)

        combined = elites + offspring
        combined.sort(key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        self.population = [sol.copy(preserve_id=False) for sol in combined[: self.population_size]]

        self._update_best_solution()
        self.iteration += 1

    @staticmethod
    def _ensure_evaluated(population: List[Solution]) -> None:
        for sol in population:
            if sol.fitness is None:
                sol.evaluate()
