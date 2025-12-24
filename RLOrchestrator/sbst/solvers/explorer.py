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
        self._last_objective_token = None

    def _maybe_reseed_for_objective(self) -> None:
        # Only available for SBSTAdapter; no-op for other problems.
        getter = getattr(self.problem, "get_active_objective_token", None)
        seed_fn = getattr(self.problem, "get_population_seeds", None)
        if not callable(getter) or not callable(seed_fn):
            return

        token = getter()
        if token is None:
            return

        if token == self._last_objective_token:
            return

        seeds = seed_fn(max_seeds=max(1, int(self.population_size // 2)))
        if not seeds:
            self._last_objective_token = token
            return

        # Build a mixed population: seeds + random fill.
        new_pop: List[Solution] = []
        for s in seeds:
            if s is None:
                continue
            clone = s.copy(preserve_id=False)
            clone.fitness = None
            new_pop.append(clone)
            if len(new_pop) >= self.population_size:
                break

        # Fill remaining slots with fresh random individuals.
        gen = getattr(self.problem, "get_initial_solution", None)
        while len(new_pop) < self.population_size and callable(gen):
            new_pop.append(gen())

        if new_pop:
            self.population = new_pop[: self.population_size]
            self._update_best_solution()

        self._last_objective_token = token

    def step(self):
        if not self.population:
            self.initialize()

        self._maybe_reseed_for_objective()
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
