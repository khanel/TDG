"""
Max-Cut specific wrappers around the generic solver catalog.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ABC import ArtificialBeeColony
from Core.problem import Solution
from GSA.gsa import GravitationalSearchAlgorithm
from HHO.hho import HarrisHawksOptimization
from LSHADE.lshade import LSHADE
from MA.memetic import MemeticAlgorithm
from MPA.mpa import MarinePredatorsAlgorithm
from SMA.sma import SlimeMouldAlgorithm
from WOA.woa import WhaleOptimizationAlgorithm


class BinaryMixin:
    """Utility mixin to convert continuous vectors into Max-Cut binary masks."""

    def _solution_from_vector(self, vector: Iterable[float]) -> Solution:
        arr = np.asarray(vector, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        mask = (arr >= 0.5).astype(int)
        sol = Solution(mask.tolist(), self.problem)
        sol.evaluate()
        return sol

    def _as_vector(self, solution: Solution) -> np.ndarray:
        rep = np.asarray(solution.representation, dtype=float)
        return rep if rep.ndim > 0 else rep.reshape(1)


class MaxCutArtificialBeeColony(ArtificialBeeColony):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutArtificialBeeColony expects a MaxCutAdapter.")
        super().__init__(problem, *args, **kwargs)


class MaxCutGravitationalSearch(BinaryMixin, GravitationalSearchAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutGravitationalSearch expects a MaxCutAdapter.")
        super().__init__(problem, *args, **kwargs)


class MaxCutHarrisHawks(BinaryMixin, HarrisHawksOptimization):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutHarrisHawks expects a MaxCutAdapter.")
        super().__init__(problem, *args, **kwargs)


class MaxCutMarinePredators(BinaryMixin, MarinePredatorsAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutMarinePredators expects a MaxCutAdapter.")
        super().__init__(problem, *args, **kwargs)


class MaxCutSlimeMould(BinaryMixin, SlimeMouldAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutSlimeMould expects a MaxCutAdapter.")
        super().__init__(problem, *args, **kwargs)


class MaxCutWhaleOptimization(BinaryMixin, WhaleOptimizationAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutWhaleOptimization expects a MaxCutAdapter.")
        super().__init__(problem, *args, **kwargs)


class MaxCutMemeticAlgorithm(MemeticAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutMemeticAlgorithm expects a MaxCutAdapter.")
        super().__init__(problem, *args, **kwargs)

    def _crossover(self, parent1: Solution, parent2: Solution):
        if self.rng.random() > self.crossover_rate:
            return [parent1.copy(preserve_id=False), parent2.copy(preserve_id=False)]
        mask1 = np.asarray(parent1.representation, dtype=int)
        mask2 = np.asarray(parent2.representation, dtype=int)
        point = int(self.rng.integers(1, mask1.size))
        child1 = np.concatenate([mask1[:point], mask2[point:]])
        child2 = np.concatenate([mask2[:point], mask1[point:]])
        return [self._make_solution(child1), self._make_solution(child2)]

    def _mutate(self, solution: Solution) -> Solution:
        mask = np.asarray(solution.representation, dtype=int).copy()
        flips = self.rng.random(mask.size) < self.mutation_rate
        if not np.any(flips):
            flips[self.rng.integers(mask.size)] = True
        mask[flips] = 1 - mask[flips]
        return self._make_solution(mask)

    def _local_search(self, solution: Solution) -> Solution:
        current = solution.copy(preserve_id=False)
        current.evaluate()
        for _ in range(self.local_search_steps):
            neighbor = self._bit_flip_neighbor(current)
            if neighbor.fitness < current.fitness:
                current = neighbor
        return current

    def _sample_neighbor(self, solution: Solution, **kwargs) -> Solution:
        return self._bit_flip_neighbor(solution)

    def _bit_flip_neighbor(self, solution: Solution) -> Solution:
        mask = np.asarray(solution.representation, dtype=int).copy()
        idx = int(self.rng.integers(mask.size))
        mask[idx] = 1 - mask[idx]
        return self._make_solution(mask)

    def _make_solution(self, mask: np.ndarray) -> Solution:
        sol = Solution(mask.tolist(), self.problem)
        sol.evaluate()
        return sol


class MaxCutLSHADE(BinaryMixin, LSHADE):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutLSHADE expects a MaxCutAdapter.")
        super().__init__(problem, *args, **kwargs)
