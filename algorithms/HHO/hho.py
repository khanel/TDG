"""
Harris Hawks Optimization (HHO) implementation.

Reference: Heidari et al., 2019 and project summary docs/candidate_HHO.md.
"""

from __future__ import annotations

from typing import Optional
import math

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

        positions_raw = [self._as_vector(sol) for sol in self.population]
        dim = max((len(p) for p in positions_raw), default=0)
        positions = np.vstack([self._resize(p, dim) for p in positions_raw])
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
                positions[i] = self._exploration_step(position, rabbit, positions, q, dim)
            else:
                r = self.rng.random()
                if r >= 0.5 and abs(E) >= 0.5:
                    positions[i] = self._soft_besiege(position, rabbit, E, J, dim)
                elif r >= 0.5 and abs(E) < 0.5:
                    positions[i] = self._hard_besiege(position, rabbit, E)
                elif r < 0.5 and abs(E) >= 0.5:
                    positions[i] = self._soft_besiege_with_dive(position, rabbit, E, J, dim)
                else:
                    positions[i] = self._hard_besiege_with_dive(position, rabbit, E, J, dim)

        positions = self._clip(positions)
        self.population = [self._solution_from_vector(vec) for vec in positions]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()
        self.iteration += 1

    # --- Phase helpers -----------------------------------------------------
    def _exploration_step(self, position, rabbit, population, q, dim: int):
        if q < 0.5:
            rand_idx = self.rng.integers(self.population_size)
            rand_hawk = self._resize(population[rand_idx], dim)
            new_pos = rand_hawk - self.rng.random(dim) * abs(rand_hawk - 2 * self.rng.random(dim) * position)
        else:
            mean_pos = np.mean(population, axis=0)
            mean_pos = self._resize(mean_pos, dim)
            rabbit = self._resize(rabbit, dim)
            new_pos = (rabbit - self.rng.random(dim) * abs(rabbit - position)) - self.rng.random(dim) * (
                mean_pos - position
            )
        return new_pos

    def _soft_besiege(self, position, rabbit, E, J, dim: int):
        rabbit = self._resize(rabbit, dim)
        distance = rabbit - position
        return distance * (self.rng.random(dim) * E) + rabbit

    def _hard_besiege(self, position, rabbit, E):
        distance = rabbit - position
        return rabbit - E * abs(distance)

    def _soft_besiege_with_dive(self, position, rabbit, E, J, dim: int):
        new_position = self._soft_besiege(position, rabbit, E, J, dim)
        levy = self._levy_flight(dim)
        y = rabbit - E * abs(J * rabbit - new_position)
        z = y + levy * self.rng.random(dim)
        return y if self._evaluate_vector(y) < self._evaluate_vector(position) else z

    def _hard_besiege_with_dive(self, position, rabbit, E, J, dim: int):
        new_position = self._hard_besiege(position, rabbit, E)
        levy = self._levy_flight(dim)
        y = rabbit - E * abs(J * rabbit - new_position)
        z = y + levy * self.rng.random(dim)
        return y if self._evaluate_vector(y) < self._evaluate_vector(position) else z

    # --- Utilities ---------------------------------------------------------
    def _levy_flight(self, dim: int | None = None):
        beta = self.levy_beta
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        d = dim if dim is not None else self.dimension
        u = self.rng.normal(0, sigma, size=d)
        v = self.rng.normal(0, 1, size=d)
        step = u / (np.abs(v) ** (1 / beta) + 1e-12)
        return step

    def _evaluate_vector(self, vector):
        # Prefer problem-specific conversion when available (e.g., permutation mapping for TSP).
        if hasattr(self, "_vector_to_permutation"):
            perm = self._vector_to_permutation(vector)
            sol = Solution(perm, self.problem)
        elif hasattr(self, "_solution_from_vector") and self._solution_from_vector.__func__ is not HarrisHawksOptimization._solution_from_vector:  # type: ignore
            sol = self._solution_from_vector(vector)  # type: ignore
        else:
            sol = Solution(np.asarray(vector, dtype=float).copy(), self.problem)
        return sol.evaluate()

    @staticmethod
    def _resize(vec: np.ndarray, dim: int) -> np.ndarray:
        arr = np.asarray(vec, dtype=float).flatten()
        if arr.size != dim:
            # Preserve order but adjust length deterministically
            base = arr.tolist()
            out = []
            while len(out) < dim:
                out.extend(base)
            return np.asarray(out[:dim], dtype=float)
        return arr

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
            if self.dimension <= 0:
                return None
            lowers = np.zeros(self.dimension, dtype=float)
            uppers = np.ones(self.dimension, dtype=float)
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
