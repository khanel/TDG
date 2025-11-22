"""
Whale Optimization Algorithm (WOA).

Implements the bubble-net feeding (encircling, spiral, random search) described
in docs/candidate_WOA.md.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


class WhaleOptimizationAlgorithm(SearchAlgorithm):
    """Swarm optimizer inspired by humpback whales' bubble-net feeding."""

    phase = "exploration"

    def __init__(
        self,
        problem,
        population_size: int,
        *,
        max_iterations: int = 500,
        b: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(problem, population_size)
        self.max_iterations = max_iterations
        self.b = b
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
        best_idx = int(np.argmin(fitness))
        best = positions[best_idx].copy()

        a = 2 - 2 * self.iteration / max(1, self.max_iterations)
        for i in range(self.population_size):
            r1 = self.rng.random()
            r2 = self.rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = self.rng.random()
            if p < 0.5:
                if abs(A) < 1:
                    distance = abs(C * best - positions[i])
                    positions[i] = best - A * distance
                else:
                    rand_idx = self.rng.integers(self.population_size)
                    rand_pos = positions[rand_idx]
                    distance = abs(C * rand_pos - positions[i])
                    positions[i] = rand_pos - A * distance
            else:
                l = self.rng.uniform(-1, 1)
                distance = abs(best - positions[i])
                positions[i] = distance * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best
            positions[i] = self._clip(positions[i])

        self.population = [self._solution_from_vector(vec) for vec in positions]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()
        self.iteration += 1

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
            # Default to unit hypercube if the adapter does not expose bounds
            if self.dimension <= 0:
                raise ValueError("WOA requires dimension > 0 to derive fallback bounds.")
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
        # Handle any mismatch between vector length and stored bounds by resizing.
        if lower.shape != vector.shape:
            lower = np.resize(lower, vector.shape)
            upper = np.resize(upper, vector.shape)
            self.bounds = (lower, upper)
        return np.clip(vector, lower, upper)

    def _as_vector(self, solution: Solution):
        arr = np.asarray(solution.representation, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    def _solution_from_vector(self, vector: np.ndarray):
        return Solution(vector.copy(), self.problem)
