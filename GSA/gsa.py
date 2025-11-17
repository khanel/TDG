"""
Gravitational Search Algorithm (GSA).

References:
- Rashedi, Nezamabadi-pour, and Saryazdi. "GSA: A Gravitational Search Algorithm."
- Candidate summary in docs/candidate_GSA.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


@dataclass
class GSAConfig:
    g0: float = 100.0
    alpha: float = 20.0
    epsilon: float = 1e-8
    k_best_ratio: float = 0.5
    max_iterations: int = 500


class GravitationalSearchAlgorithm(SearchAlgorithm):
    """Continuous optimization solver using gravitational interactions among agents."""

    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        config: Optional[GSAConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(problem, population_size)
        self.config = config or GSAConfig()
        self.rng = np.random.default_rng(seed)
        self.velocities: Optional[np.ndarray] = None
        self.bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.dimension: int = 0

    def initialize(self):
        super().initialize()
        self.dimension = self._infer_dimension()
        self.velocities = np.zeros((self.population_size, self.dimension), dtype=float)
        self.bounds = self._extract_bounds()
        self.iteration = 0

    def step(self):
        if not self.population:
            self.initialize()
            if not self.population:
                return

        positions = np.array([self._as_vector(sol) for sol in self.population], dtype=float)
        fitness = np.array([sol.fitness if sol.fitness is not None else sol.evaluate() for sol in self.population], dtype=float)

        best_f = np.min(fitness)
        worst_f = np.max(fitness)
        if np.isclose(best_f, worst_f):
            masses = np.ones_like(fitness)
        else:
            masses = (worst_f - fitness) / (worst_f - best_f + self.config.epsilon)
        masses = masses / (np.sum(masses) + self.config.epsilon)

        g = self._gravitational_constant()
        k_best = max(1, int(self.config.k_best_ratio * self.population_size))
        best_indices = np.argsort(fitness)[:k_best]

        new_positions = positions.copy()
        for i in range(self.population_size):
            force = np.zeros(self.dimension, dtype=float)
            for j in best_indices:
                if i == j:
                    continue
                direction = positions[j] - positions[i]
                distance = np.linalg.norm(direction) + self.config.epsilon
                force += self.rng.random(self.dimension) * (
                    g * masses[j] * direction / distance
                )
            acceleration = force / (masses[i] + self.config.epsilon)
            self.velocities[i] = self.rng.random(self.dimension) * self.velocities[i] + acceleration
            new_positions[i] = positions[i] + self.velocities[i]

        new_positions = self._clip(new_positions)
        self.population = [self._solution_from_vector(vec) for vec in new_positions]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()
        self.iteration += 1

    def _gravitational_constant(self) -> float:
        cfg = self.config
        progress = min(1.0, self.iteration / max(1, cfg.max_iterations))
        return cfg.g0 * np.exp(-cfg.alpha * progress)

    def _infer_dimension(self) -> int:
        if not self.population:
            return 0
        rep = self.population[0].representation
        if hasattr(rep, "__len__"):
            return len(rep)
        return 1

    def _as_vector(self, solution: Solution) -> np.ndarray:
        rep = solution.representation
        arr = np.asarray(rep, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    def _solution_from_vector(self, vector: np.ndarray) -> Solution:
        return Solution(vector.copy(), self.problem)

    def _extract_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
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

    def _clip(self, positions: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return positions
        lower, upper = self.bounds
        return np.clip(positions, lower, upper)
