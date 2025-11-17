"""
Harris Hawks Optimization (HHO) implementation.

Reference: Heidari et al., 2019 and project summary docs/candidate_HHO.md.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


class HarrisHawksOptimization(SearchAlgorithm):
    """Population-based swarm inspired by Harris hawks hunting strategy."""

    phase = "exploration"

    def __init__(
        self,
        problem,
        population_size: int,
        *,
        max_iterations: int = 500,
        levy_beta: float = 1.5,
        seed: Optional[int] = None,
    ):
        super().__init__(problem, population_size)
        self.max_iterations = max_iterations
        self.levy_beta = levy_beta
        self.rng = np.random.default_rng(seed)
        self.bounds = None
        self.dimension = 0

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
        rabbit = positions[best_idx].copy()
        rabbit_fit = fitness[best_idx]

        for i in range(self.population_size):
            position = positions[i]
            E0 = self.rng.uniform(-1, 1)
            E = 2 * E0 * (1 - self.iteration / max(1, self.max_iterations))
            J = 2 * (1 - self.rng.random())
            q = self.rng.random()

            if abs(E) >= 1:
                positions[i] = self._exploration_step(position, rabbit, positions, q)
            else:
                r = self.rng.random()
                if r >= 0.5 and abs(E) >= 0.5:
                    positions[i] = self._soft_besiege(position, rabbit, E, J)
                elif r >= 0.5 and abs(E) < 0.5:
                    positions[i] = self._hard_besiege(position, rabbit, E)
                elif r < 0.5 and abs(E) >= 0.5:
                    positions[i] = self._soft_besiege_with_dive(position, rabbit, E, J)
                else:
                    positions[i] = self._hard_besiege_with_dive(position, rabbit, E, J)

        positions = self._clip(positions)
        self.population = [self._solution_from_vector(vec) for vec in positions]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()
        self.iteration += 1

    # --- Phase helpers -----------------------------------------------------
    def _exploration_step(self, position, rabbit, population, q):
        if q < 0.5:
            rand_idx = self.rng.integers(self.population_size)
            rand_hawk = population[rand_idx]
            new_pos = rand_hawk - self.rng.random(self.dimension) * abs(rand_hawk - 2 * self.rng.random(self.dimension) * position)
        else:
            mean_pos = np.mean(population, axis=0)
            new_pos = (rabbit - self.rng.random(self.dimension) * abs(rabbit - position)) - self.rng.random(self.dimension) * (
                mean_pos - position
            )
        return new_pos

    def _soft_besiege(self, position, rabbit, E, J):
        distance = rabbit - position
        return distance * (self.rng.random(self.dimension) * E) + rabbit

    def _hard_besiege(self, position, rabbit, E):
        distance = rabbit - position
        return rabbit - E * abs(distance)

    def _soft_besiege_with_dive(self, position, rabbit, E, J):
        new_position = self._soft_besiege(position, rabbit, E, J)
        levy = self._levy_flight()
        y = rabbit - E * abs(J * rabbit - new_position)
        z = y + levy * self.rng.random(self.dimension)
        return y if self._evaluate_vector(y) < self._evaluate_vector(position) else z

    def _hard_besiege_with_dive(self, position, rabbit, E, J):
        new_position = self._hard_besiege(position, rabbit, E)
        levy = self._levy_flight()
        y = rabbit - E * abs(J * rabbit - new_position)
        z = y + levy * self.rng.random(self.dimension)
        return y if self._evaluate_vector(y) < self._evaluate_vector(position) else z

    # --- Utilities ---------------------------------------------------------
    def _levy_flight(self):
        beta = self.levy_beta
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = self.rng.normal(0, sigma, size=self.dimension)
        v = self.rng.normal(0, 1, size=self.dimension)
        step = u / (np.abs(v) ** (1 / beta) + 1e-12)
        return step

    def _evaluate_vector(self, vector):
        sol = Solution(vector.copy(), self.problem)
        return sol.evaluate()

    def _clip(self, positions):
        if self.bounds is None:
            return positions
        lower, upper = self.bounds
        return np.clip(positions, lower, upper)

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
        if lowers is None and uppers is None:
            return None
        if lowers is None:
            lowers = np.full(self.dimension, -np.inf)
        if uppers is None:
            uppers = np.full(self.dimension, np.inf)
        if lowers.size == 1:
            lowers = np.full(self.dimension, lowers.item())
        if uppers.size == 1:
            uppers = np.full(self.dimension, uppers.item())
        return lowers.astype(float), uppers.astype(float)

    def _as_vector(self, solution: Solution) -> np.ndarray:
        arr = np.asarray(solution.representation, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    def _solution_from_vector(self, vector: np.ndarray) -> Solution:
        return Solution(vector.copy(), self.problem)
