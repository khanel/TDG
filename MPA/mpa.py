"""
Marine Predators Algorithm (MPA).

Follows the multi-phase Brownian / Lévy motion described in the original
paper and summarized in docs/candidate_MPA.md.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


class MarinePredatorsAlgorithm(SearchAlgorithm):
    """Three-phase predator optimization strategy."""

    phase = "exploration"

    def __init__(
        self,
        problem,
        population_size: int,
        *,
        max_iterations: int = 500,
        fad_probability: float = 0.2,
        seed: Optional[int] = None,
    ):
        super().__init__(problem, population_size)
        self.max_iterations = max_iterations
        self.fad_probability = fad_probability
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
        CF = np.exp((self.iteration / max(1, self.max_iterations)) - 1)  # control factor
        ratio = self.iteration / max(1, self.max_iterations)

        for i in range(self.population_size):
            RB = self.rng.normal(0, 1, size=self.dimension)
            RL = self._levy()
            if ratio < 1 / 3:
                positions[i] = positions[i] + RB * (best - RB * positions[i])
            elif ratio < 2 / 3:
                positions[i] = best + RB * (positions[i] - best)
            else:
                positions[i] = best + RL * (best - positions[i])

            if self.rng.random() < self.fad_probability:
                jump = (self.bounds[1] - self.bounds[0]) * self.rng.random(self.dimension) + self.bounds[0]
                positions[i] = positions[i] + CF * (jump - positions[i])

            positions[i] = self._clip(positions[i])

        self.population = [self._solution_from_vector(vec) for vec in positions]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()
        self.iteration += 1

    def _levy(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = self.rng.normal(0, sigma, size=self.dimension)
        v = self.rng.normal(0, 1, size=self.dimension)
        return u / (np.abs(v) ** (1 / beta) + 1e-9)

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
            raise ValueError("MPA requires finite bounds for initialization.")
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
