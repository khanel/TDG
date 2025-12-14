"""
Knapsack-specific wrappers around the generic solver catalog.

Each wrapper ensures solutions remain binary/feasible by thresholding and
repairing masks via the adapter before evaluation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from algorithms.ABC import ArtificialBeeColony
from Core.problem import Solution
from algorithms.GSA import GravitationalSearchAlgorithm
from algorithms.HHO import HarrisHawksOptimization
from algorithms.LSHADE import LSHADE
from algorithms.MA import MemeticAlgorithm
from algorithms.MPA import MarinePredatorsAlgorithm
from algorithms.SMA import SlimeMouldAlgorithm
from algorithms.WOA import WhaleOptimizationAlgorithm


class BinarySolutionMixin:
    """Utility helpers for converting real-valued vectors into binary masks."""

    def _binarize_vector(self, vector: Iterable[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        mask = (arr >= 0.5).astype(int)
        repair = getattr(self.problem, "repair_mask", None)
        if callable(repair):
            mask = np.asarray(repair(mask.tolist()), dtype=int)
        return mask

    def _solution_from_vector(self, vector: Iterable[float]) -> Solution:
        mask = self._binarize_vector(vector)
        return Solution(mask.tolist(), self.problem)


class KnapsackArtificialBeeColony(ArtificialBeeColony):
    """ABC already respects discrete representations; wrapper ensures adapter presence."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackArtificialBeeColony expects a KnapsackAdapter.")
        super().__init__(problem, *args, **kwargs)


class KnapsackGravitationalSearch(BinarySolutionMixin, GravitationalSearchAlgorithm):
    """Binary-aware GSA variant."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackGravitationalSearch expects a KnapsackAdapter.")
        super().__init__(problem, *args, **kwargs)


class KnapsackHarrisHawks(BinarySolutionMixin, HarrisHawksOptimization):
    """Binary-aware HHO variant."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackHarrisHawks expects a KnapsackAdapter.")
        super().__init__(problem, *args, **kwargs)


class KnapsackMarinePredators(BinarySolutionMixin, MarinePredatorsAlgorithm):
    """Binary-aware MPA variant."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackMarinePredators expects a KnapsackAdapter.")
        super().__init__(problem, *args, **kwargs)


class KnapsackSlimeMould(BinarySolutionMixin, SlimeMouldAlgorithm):
    """Binary-aware SMA variant."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackSlimeMould expects a KnapsackAdapter.")
        super().__init__(problem, *args, **kwargs)


class KnapsackWhaleOptimization(BinarySolutionMixin, WhaleOptimizationAlgorithm):
    """Binary-aware WOA variant."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackWhaleOptimization expects a KnapsackAdapter.")
        super().__init__(problem, *args, **kwargs)


class KnapsackLSHADE(BinarySolutionMixin, LSHADE):
    """Binary Success-History Adaptive Differential Evolution."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackLSHADE expects a KnapsackAdapter.")
        super().__init__(problem, *args, **kwargs)

    def _mutate(self, idx, F, positions, fitness):
        vector = super()._mutate(idx, F, positions, fitness)
        return self._binarize_vector(vector)

    def _crossover(self, target, donor, CR):
        vector = super()._crossover(target, donor, CR)
        return self._binarize_vector(vector)


class KnapsackMemeticAlgorithm(MemeticAlgorithm):
    """Memetic GA tuned for binary masks."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "knapsack_problem"):
            raise ValueError("KnapsackMemeticAlgorithm expects a KnapsackAdapter.")
        super().__init__(problem, *args, **kwargs)

    def _solution_from_mask(self, mask: np.ndarray) -> Solution:
        mask = mask.astype(int)
        repair = getattr(self.problem, "repair_mask", None)
        if callable(repair):
            mask = np.asarray(repair(mask.tolist()), dtype=int)
        sol = Solution(mask.tolist(), self.problem)
        sol.evaluate()
        return sol

    def _crossover(self, parent1: Solution, parent2: Solution):
        if self.rng.random() > self.crossover_rate:
            return [parent1.copy(preserve_id=False), parent2.copy(preserve_id=False)]
        mask1 = np.asarray(parent1.representation, dtype=int)
        mask2 = np.asarray(parent2.representation, dtype=int)
        point = int(self.rng.integers(1, mask1.size))
        child1 = np.concatenate([mask1[:point], mask2[point:]])
        child2 = np.concatenate([mask2[:point], mask1[point:]])
        return [self._solution_from_mask(child1), self._solution_from_mask(child2)]

    def _mutate(self, solution: Solution) -> Solution:
        mask = np.asarray(solution.representation, dtype=int)
        flips = self.rng.random(mask.size) < self.mutation_rate
        if not np.any(flips):
            flips[self.rng.integers(mask.size)] = True
        mutated = mask.copy()
        mutated[flips] = 1 - mutated[flips]
        return self._solution_from_mask(mutated)

    def _local_search(self, solution: Solution) -> Solution:
        current = solution.copy(preserve_id=False)
        current.evaluate()
        for _ in range(self.local_search_steps):
            neighbor = self._bitflip_neighbor(current)
            if neighbor.fitness < current.fitness:
                current = neighbor
        return current

    def _sample_neighbor(self, solution: Solution, **kwargs) -> Solution:
        return self._bitflip_neighbor(solution)

    def _bitflip_neighbor(self, solution: Solution) -> Solution:
        mask = np.asarray(solution.representation, dtype=int)
        idx = int(self.rng.integers(mask.size))
        neighbor = mask.copy()
        neighbor[idx] = 1 - neighbor[idx]
        return self._solution_from_mask(neighbor)
