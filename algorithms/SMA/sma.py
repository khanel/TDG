"""
Slime Mould Algorithm (SMA) implementation.

Captures the propagation-contraction dynamics summarized in
docs/candidate_SMA.md.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


class SlimeMouldAlgorithm(SearchAlgorithm):
    """SMA with oscillation-based weighting and pheromone-vibration modeling."""

    phase = "exploration"

    def __init__(
        self,
        problem,
        population_size: int,
        *,
        max_iterations: int = 500,
        seed: Optional[int] = None,
    ):
        super().__init__(problem, population_size)
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        self.dimension = 0
        self.bounds = None

    def initialize(self):
        super().initialize()
        self.dimension = self._infer_dimension()
        self.bounds = self._extract_bounds()
        self.iteration = 0

    def step(self):
        if not self.population:
            self.initialize()
            if not self.population:
                return

        positions = np.array([self._as_vector(sol) for sol in self.population], dtype=float)
        fitness = np.array([sol.fitness if sol.fitness is not None else sol.evaluate() for sol in self.population])
        idx_sorted = np.argsort(fitness)
        best = positions[idx_sorted[0]]
        worst = positions[idx_sorted[-1]]
        best_fit = fitness[idx_sorted[0]]
        worst_fit = fitness[idx_sorted[-1]]

        z = (self.iteration + 1) / max(1, self.max_iterations)
        vb = self.rng.random(self.dimension)

        new_positions = positions.copy()
        for rank, idx in enumerate(idx_sorted):
            weight = self._compute_weight(fitness[idx], best_fit, worst_fit)
            if self.rng.random() < self.rng.random():
                random_idx = self.rng.integers(self.population_size)
                new_positions[idx] = best + self.rng.random(self.dimension) * weight * (
                    self.bounds[0] + self.rng.random(self.dimension) * (self.bounds[1] - self.bounds[0])
                )
            else:
                j, k = self.rng.choice(self.population_size, size=2, replace=False)
                new_positions[idx] = positions[idx] + weight * (positions[j] - positions[k])
            new_positions[idx] = self._clip(new_positions[idx])

        self.population = [self._solution_from_vector(vec) for vec in new_positions]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()
        self.iteration += 1

    def _compute_weight(self, fitness, best_fit, worst_fit):
        if np.isclose(best_fit, worst_fit):
            return self.rng.random(self.dimension)
        norm = (fitness - best_fit) / (worst_fit - best_fit + 1e-12)
        return np.tanh(norm - 0.5)

    def _infer_dimension(self):
        if not self.population:
            return 0
        rep = self.population[0].representation
        if hasattr(rep, "__len__"):
            return len(rep)
        return 1

    def _extract_bounds(self):
        try:
            info = self.problem.get_problem_info()
        except Exception:
            info = {}
        lowers = np.asarray(info.get("lower_bounds")) if "lower_bounds" in info else None
        uppers = np.asarray(info.get("upper_bounds")) if "upper_bounds" in info else None
        if lowers is None or uppers is None:
            # Default to unit hypercube bounds when adapters omit them
            if self.dimension <= 0:
                raise ValueError("SMA requires dimension > 0 to derive fallback bounds.")
            lowers = np.zeros(self.dimension, dtype=float)
            uppers = np.ones(self.dimension, dtype=float)
        if lowers.size == 1:
            lowers = np.full(self.dimension, lowers.item())
        if uppers.size == 1:
            uppers = np.full(self.dimension, uppers.item())
        return lowers.astype(float), uppers.astype(float)

    def _clip(self, vector):
        if self.bounds is None:
            return vector
        lower, upper = self.bounds
        return np.clip(vector, lower, upper)

    def _as_vector(self, solution: Solution):
        arr = np.asarray(solution.representation, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    def _solution_from_vector(self, vector: np.ndarray):
        return Solution(vector.copy(), self.problem)
