"""
Specialized MAP-Elites style exploration solver for TSP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from Core.problem import Solution, ProblemInterface
from Core.search_algorithm import SearchAlgorithm


@dataclass
class ArchiveEntry:
    solution: Solution
    descriptor: np.ndarray


class TSPMapElites(SearchAlgorithm):
    """MAP-Elites-inspired explorer tailored for TSP problems."""

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 64,
        *,
        bins_per_dim: Tuple[int, int] = (16, 16),
        random_injection_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "tsp_problem"):
            raise ValueError("TSPMapElites expects a TSPAdapter exposing `tsp_problem`.")
        super().__init__(problem, population_size)
        if len(bins_per_dim) != 2:
            raise ValueError("bins_per_dim must contain exactly two entries.")
        self.bins_per_dim = tuple(max(2, int(b)) for b in bins_per_dim)
        self.random_injection_rate = float(np.clip(random_injection_rate, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.archive: Dict[Tuple[int, int], ArchiveEntry] = {}
        self._refresh_distance_cache()

    def initialize(self):
        self._refresh_distance_cache()
        self.archive.clear()
        super().initialize()
        for sol in self.population:
            self._evaluate_if_needed(sol)
            self._try_insert(sol)
        self._refresh_population()
        self._update_best_solution()

    def step(self):
        if not self.archive:
            self.initialize()
            self.iteration += 1
            return

        parent = self._random_sample() if self.rng.random() < self.random_injection_rate else self._select_parent()
        child = self._mutate(parent)
        self._evaluate_if_needed(child)
        self._try_insert(child)

        self._refresh_population()
        self._update_best_solution()
        self.iteration += 1

    # Helpers -----------------------------------------------------------
    def _refresh_distance_cache(self):
        tsp = self.problem.tsp_problem
        self.distance_matrix = np.asarray(tsp.cities_graph.get_weights(), dtype=float)
        self.num_cities = self.distance_matrix.shape[0]
        self.max_edge = float(np.max(self.distance_matrix)) if self.distance_matrix.size else 1.0

    @staticmethod
    def _evaluate_if_needed(sol: Solution) -> None:
        if sol.fitness is None:
            sol.evaluate()

    def _descriptor(self, sol: Solution) -> np.ndarray:
        rep = list(sol.representation)
        if len(rep) != self.num_cities:
            raise ValueError("Tour representation length mismatch.")
        indices = [idx - 1 for idx in rep]
        indices.append(indices[0])
        edges = [self.distance_matrix[indices[i], indices[i + 1]] for i in range(self.num_cities)]
        edges = np.asarray(edges, dtype=float)
        mean = float(np.mean(edges)) / max(1e-9, self.max_edge)
        std = float(np.std(edges)) / max(1e-9, self.max_edge)
        return np.clip(np.array([mean, std], dtype=float), 0.0, 1.0)

    def _cell_key(self, descriptor: np.ndarray) -> Tuple[int, int]:
        scaled = descriptor * (np.asarray(self.bins_per_dim, dtype=float) - 1.0)
        indices = np.floor(scaled + 1e-9).astype(int)
        indices = np.clip(indices, 0, np.asarray(self.bins_per_dim) - 1)
        return tuple(int(i) for i in indices)

    def _try_insert(self, sol: Solution) -> None:
        descriptor = self._descriptor(sol)
        key = self._cell_key(descriptor)
        current = self.archive.get(key)
        if current is None or (sol.fitness is not None and sol.fitness < current.solution.fitness):
            self.archive[key] = ArchiveEntry(solution=sol.copy(preserve_id=False), descriptor=descriptor)

    def _select_parent(self) -> Solution:
        entries = list(self.archive.values())
        idx = int(self.rng.integers(len(entries)))
        return entries[idx].solution

    def _random_sample(self) -> Solution:
        sol = self.problem.get_initial_solution()
        self._evaluate_if_needed(sol)
        return sol

    def _mutate(self, parent: Solution) -> Solution:
        rep = list(parent.representation)
        if len(rep) < 4:
            i, j = sorted(self.rng.choice(len(rep), size=2, replace=False))
        else:
            i, j = sorted(self.rng.choice(range(1, len(rep)), size=2, replace=False))
        rep[i], rep[j] = rep[j], rep[i]
        return Solution(rep, self.problem)

    def _refresh_population(self) -> None:
        if not self.archive:
            return
        elites = sorted((entry.solution for entry in self.archive.values()), key=lambda s: s.fitness or float("inf"))
        self.population = [sol.copy(preserve_id=False) for sol in elites[: self.population_size]]
        if not self.population:
            self.population = [self.problem.get_initial_solution()]
