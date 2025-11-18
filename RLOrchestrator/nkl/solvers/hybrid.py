"""
NK-Landscape wrappers for the generic solver catalog.
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


class BinaryMaskMixin:
    """Provides conversions between continuous vectors and binary masks."""

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


class NKLArtificialBeeColony(ArtificialBeeColony):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "nkl_problem"):
            raise ValueError("NKLArtificialBeeColony expects an NKLAdapter.")
        super().__init__(problem, *args, **kwargs)


class NKLGravitationalSearch(BinaryMaskMixin, GravitationalSearchAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "nkl_problem"):
            raise ValueError("NKLGravitationalSearch expects an NKLAdapter.")
        super().__init__(problem, *args, **kwargs)


class NKLHarrisHawks(BinaryMaskMixin, HarrisHawksOptimization):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "nkl_problem"):
            raise ValueError("NKLHarrisHawks expects an NKLAdapter.")
        super().__init__(problem, *args, **kwargs)


class NKLMarinePredators(BinaryMaskMixin, MarinePredatorsAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "nkl_problem"):
            raise ValueError("NKLMarinePredators expects an NKLAdapter.")
        super().__init__(problem, *args, **kwargs)


class NKLSlimeMould(BinaryMaskMixin, SlimeMouldAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "nkl_problem"):
            raise ValueError("NKLSlimeMould expects an NKLAdapter.")
        super().__init__(problem, *args, **kwargs)


class NKLWhaleOptimization(BinaryMaskMixin, WhaleOptimizationAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "nkl_problem"):
            raise ValueError("NKLWhaleOptimization expects an NKLAdapter.")
        super().__init__(problem, *args, **kwargs)


class NKLMemeticAlgorithm(MemeticAlgorithm):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "nkl_problem"):
            raise ValueError("NKLMemeticAlgorithm expects an NKLAdapter.")
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


class NKLLSHADE(BinaryMaskMixin, LSHADE):
    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "nkl_problem"):
            raise ValueError("NKLLSHADE expects an NKLAdapter.")
        super().__init__(problem, *args, **kwargs)
