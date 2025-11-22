"""
Permutation-aware wrappers around the generic solver catalog for TSP.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from ABC import ArtificialBeeColony
from GSA.gsa import GravitationalSearchAlgorithm
from HHO.hho import HarrisHawksOptimization
from LSHADE.lshade import LSHADE
from MA.memetic import MemeticAlgorithm
from MPA.mpa import MarinePredatorsAlgorithm
from SMA.sma import SlimeMouldAlgorithm
from WOA.woa import WhaleOptimizationAlgorithm
from Core.problem import Solution


class TSPPermutationMixin:
    """Converts between permutation tours and continuous vectors via random keys."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "tsp_problem"):
            raise ValueError(f"{self.__class__.__name__} expects a TSPAdapter exposing `tsp_problem`.")
        self._tsp_vertices: List[int] | None = None
        super().__init__(problem, *args, **kwargs)

    # Vector helpers --------------------------------------------------------
    def _as_vector(self, solution: Solution) -> np.ndarray:
        return self._permutation_to_vector(list(solution.representation))

    def _solution_from_vector(self, vector: Iterable[float]) -> Solution:
        tour = self._vector_to_permutation(vector)
        return Solution(tour, self.problem)

    def _permutation_to_vector(self, tour: List[int]) -> np.ndarray:
        vertices = self._get_vertices()
        positions = {city: idx for idx, city in enumerate(tour)}
        denom = max(1, len(vertices) - 1)
        vector = np.array([positions.get(city, 0) / denom for city in vertices], dtype=float)
        return vector

    def _vector_to_permutation(self, vector: Iterable[float]) -> List[int]:
        arr = np.asarray(vector, dtype=float).flatten()
        vertices = self._get_vertices()
        if arr.size != len(vertices):
            arr = np.resize(arr, len(vertices))
        order = np.argsort(arr, kind="mergesort")
        perm = [vertices[idx] for idx in order]
        first = vertices[0]
        idx = perm.index(first)
        if idx != 0:
            perm = perm[idx:] + perm[:idx]
        return perm

    def _get_vertices(self) -> List[int]:
        graph = self.problem.tsp_problem.cities_graph
        vertices = list(graph.get_vertices())
        if self._tsp_vertices is None or len(self._tsp_vertices) != len(vertices):
            self._tsp_vertices = vertices
        return self._tsp_vertices


class TSPArtificialBeeColony(ArtificialBeeColony):
    """ABC configured for TSP adapters."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "tsp_problem"):
            raise ValueError("TSPArtificialBeeColony expects a TSPAdapter.")
        super().__init__(problem, *args, **kwargs)


class TSPGravitationalSearch(TSPPermutationMixin, GravitationalSearchAlgorithm):
    """Random-key GSA."""

    pass


class TSPHarrisHawks(TSPPermutationMixin, HarrisHawksOptimization):
    """Random-key HHO."""

    pass


class TSPMarinePredators(TSPPermutationMixin, MarinePredatorsAlgorithm):
    """Random-key MPA."""

    pass


class TSPSlimeMould(TSPPermutationMixin, SlimeMouldAlgorithm):
    """Random-key SMA."""

    pass


class TSPWhaleOptimization(TSPPermutationMixin, WhaleOptimizationAlgorithm):
    """Random-key WOA."""

    pass


class TSPMemeticAlgorithm(MemeticAlgorithm):
    """Permutation-aware memetic algorithm with order-based operators."""

    def __init__(self, problem, *args, **kwargs):
        if not hasattr(problem, "tsp_problem"):
            raise ValueError("TSPMemeticAlgorithm expects a TSPAdapter.")
        super().__init__(problem, *args, **kwargs)

    def _crossover(self, parent1: Solution, parent2: Solution):
        if self.rng.random() > self.crossover_rate:
            return [parent1.copy(preserve_id=False), parent2.copy(preserve_id=False)]
        tour1 = list(parent1.representation)
        tour2 = list(parent2.representation)
        length = len(tour1)
        a, b = sorted(self.rng.choice(range(1, length), size=2, replace=False))
        child = [None] * length
        child[0] = tour1[0]
        child[a:b] = tour1[a:b]
        idx = b
        for city in tour2:
            if city in child:
                continue
            if idx >= length:
                idx = 1  # keep start city fixed
            if child[idx] is None:
                child[idx] = city
                idx += 1
        child = self._repair_child(child, tour2)

        child2 = [None] * length
        child2[0] = tour2[0]
        child2[a:b] = tour2[a:b]
        idx = b
        for city in tour1:
            if city in child2:
                continue
            if idx >= length:
                idx = 1
            if child2[idx] is None:
                child2[idx] = city
                idx += 1
        child2 = self._repair_child(child2, tour1)

        return [self._make_solution(child), self._make_solution(child2)]

    def _repair_child(self, child: List[int], fallback: List[int]) -> List[int]:
        missing = [city for city in fallback if city not in child]
        out = child[:]
        idx = 1
        for city in missing:
            while idx < len(out) and out[idx] is not None:
                idx += 1
            if idx >= len(out):
                break
            out[idx] = city
        return [c if c is not None else fallback[0] for c in out]

    def _mutate(self, solution: Solution) -> Solution:
        tour = list(solution.representation)
        if len(tour) <= 3:
            solution.evaluate()
            return solution
        if self.rng.random() > self.mutation_rate:
            solution.evaluate()
            return solution
        i, j = sorted(self.rng.choice(range(1, len(tour)), size=2, replace=False))
        tour[i], tour[j] = tour[j], tour[i]
        return self._make_solution(tour)

    def _local_search(self, solution: Solution) -> Solution:
        current = solution.copy(preserve_id=False)
        current.evaluate()
        for _ in range(self.local_search_steps):
            neighbor = self._sample_neighbor(current)
            if neighbor.fitness < current.fitness:
                current = neighbor
        return current

    def _sample_neighbor(self, solution: Solution, k: int = 1) -> Solution:
        if hasattr(self.problem, "sample_neighbors"):
            neighbors = self.problem.sample_neighbors(solution, max(1, k))
            if neighbors:
                best = min(neighbors, key=lambda sol: sol.evaluate())
                return best
        tour = list(solution.representation)
        i, j = sorted(self.rng.choice(range(1, len(tour)), size=2, replace=False))
        tour[i:j] = reversed(tour[i:j])
        return self._make_solution(tour)

    def _make_solution(self, tour: List[int]) -> Solution:
        sol = Solution(tour, self.problem)
        sol.evaluate()
        return sol


class TSPLSHADE(TSPPermutationMixin, LSHADE):
    """Permutation-aware LSHADE with random-key embedding."""

    def step(self):
        if not self.population:
            self.initialize()
            if not self.population:
                return

        cfg = self.config
        positions = np.array([self._as_vector(sol) for sol in self.population], dtype=float)
        fitness = np.array([sol.fitness if sol.fitness is not None else sol.evaluate() for sol in self.population], dtype=float)
        pop_size = positions.shape[0]
        next_positions = positions.copy()
        next_solutions = [sol.copy(preserve_id=False) for sol in self.population]
        SF: List[float] = []
        SCR: List[float] = []
        delta_f: List[float] = []

        for i in range(pop_size):
            Fi, CRi = self._sample_parameters()
            donor = self._mutate(i, Fi, positions, fitness)
            trial = self._crossover(positions[i], donor, CRi)
            trial = self._clip(trial)
            trial_solution = self._solution_from_vector(trial)
            trial_fit = trial_solution.evaluate()
            if trial_fit <= fitness[i]:
                next_positions[i] = trial
                next_solutions[i] = trial_solution
                SF.append(Fi)
                SCR.append(CRi)
                delta_f.append(abs(trial_fit - fitness[i]))
                self.archive.append(positions[i])
            else:
                next_positions[i] = positions[i]
                next_solutions[i] = self.population[i]

        if SF:
            self._update_history(SF, SCR, delta_f)

        target_size = self._current_population_size()
        if next_positions.shape[0] > target_size:
            sorted_idx = np.argsort([sol.fitness if sol.fitness is not None else sol.evaluate() for sol in next_solutions])
            keep_idx = sorted_idx[:target_size]
            next_positions = next_positions[keep_idx]
            next_solutions = [next_solutions[idx] for idx in keep_idx]

        self.population = [sol.copy(preserve_id=False) for sol in next_solutions]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()
        self.iteration += 1
