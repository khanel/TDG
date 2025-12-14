"""SBST exploration solver (placeholder)."""

from __future__ import annotations

from typing import List

import numpy as np

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


class SBSTRandomExplorer(SearchAlgorithm):
    """Simple random mutation explorer for the SBST scaffold."""

    phase = "exploration"

    def __init__(self, problem, population_size: int = 48, *, mutation_rate: float = 0.35, seed: int | None = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self._mutation_rate = float(mutation_rate)
        self._rng = np.random.default_rng(seed)

    def step(self):
        if not self.population:
            self.initialize()

        self.ensure_population_evaluated()

        new_pop: List[Solution] = []
        for sol in self.population:
            child = sol.copy(preserve_id=False)
            rep = child.representation
            genes = list(rep.get("genes", [])) if isinstance(rep, dict) else list(rep)
            if genes:
                for i in range(len(genes)):
                    if float(self._rng.random()) < self._mutation_rate:
                        genes[i] = int(self._rng.integers(0, 100))
            child.representation = {"genes": genes}
            child.fitness = None
            child.evaluate()
            new_pop.append(child)

        self.population = new_pop
        self._update_best_solution()
        self.iteration += 1
