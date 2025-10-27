"""
TSP problem adapter for RLOrchestrator.
"""

from typing import Dict, Any, List, Optional, Iterable

import numpy as np

from TSP.TSP import TSPProblem, Graph
from Core.problem import Solution, ProblemInterface


class TSPAdapter(ProblemInterface):
    """Adapter for TSP problems with optional per-episode randomization."""

    def __init__(
        self,
        tsp_problem: Optional[TSPProblem] = None,
        *,
        num_cities: int = 20,
        seed: Optional[int] = 42,
        grid_size: float = 100.0,
        coords: Optional[Iterable[Iterable[float]]] = None,
        distance_matrix: Optional[Iterable[Iterable[float]]] = None,
    ):
        self._grid_size = float(grid_size)
        self._base_seed = seed
        self._coords_source = np.asarray(coords, dtype=float) if coords is not None else None
        self._dist_source = np.asarray(distance_matrix, dtype=float) if distance_matrix is not None else None
        self._randomizable = tsp_problem is None and self._coords_source is None and self._dist_source is None
        self._rng = np.random.default_rng(seed)

        if isinstance(num_cities, (tuple, list)) and len(num_cities) == 2:
            lo_val = int(num_cities[0])
            hi_val = int(num_cities[1])
            low, high = sorted((lo_val, hi_val))
            self._num_cities_range: Optional[tuple[int, int]] = (max(3, low), max(3, high))
            self._num_cities_fixed: Optional[int] = None
        else:
            self._num_cities_range = None
            self._num_cities_fixed = max(3, int(num_cities))

        if tsp_problem is not None:
            self.tsp_problem = tsp_problem
            self._bounds = self._estimate_bounds_from_graph(tsp_problem)
        else:
            self._rebuild_problem(self._coords_source, self._dist_source)

    def evaluate(self, solution: Solution) -> float:
        return self.tsp_problem.evaluate(solution)

    def get_initial_solution(self) -> Solution:
        return self.tsp_problem.get_initial_solution()

    def get_initial_population(self, size: int) -> List[Solution]:
        return [self.get_initial_solution() for _ in range(size)]

    def get_problem_info(self) -> Dict[str, Any]:
        info = self.tsp_problem.get_problem_info()
        info["problem_type"] = "permutation"
        return info

    def get_bounds(self) -> Dict[str, float]:
        return dict(self._bounds)

    def regenerate_instance(self) -> bool:
        if not self._randomizable or self._rng is None:
            return False
        n = self._sample_num_cities(self._rng)
        coords = self._rng.random((n, 2)) * self._grid_size
        dist = self._pairwise_distances(coords)
        self._rebuild_problem(coords, dist)
        return True

    def _rebuild_problem(self, coords: Optional[Iterable[Iterable[float]]], dist: Optional[Iterable[Iterable[float]]]) -> None:
        coords_array: Optional[np.ndarray] = None
        dist_array: Optional[np.ndarray] = None

        if coords is not None:
            coords_array = np.asarray(coords, dtype=float)
            if coords_array.ndim != 2 or coords_array.shape[1] != 2:
                raise ValueError("coords must have shape (N, 2)")
        if dist is not None:
            dist_array = np.asarray(dist, dtype=float)
            if dist_array.ndim != 2 or dist_array.shape[0] != dist_array.shape[1]:
                raise ValueError("distance_matrix must be square")

        if coords_array is None and dist_array is None:
            rng = self._rng if self._rng is not None else np.random.default_rng()
            n = self._sample_num_cities(rng)
            coords_array = rng.random((n, 2)) * self._grid_size
            dist_array = self._pairwise_distances(coords_array)
        elif coords_array is None and dist_array is not None:
            coords_array = np.zeros((dist_array.shape[0], 2), dtype=float)
        elif coords_array is not None and dist_array is None:
            dist_array = self._pairwise_distances(coords_array)

        if coords_array is None or dist_array is None:
            raise ValueError("Unable to build TSP problem: insufficient data.")
        if coords_array.shape[0] != dist_array.shape[0]:
            raise ValueError("coords and distance_matrix size mismatch")

        np.fill_diagonal(dist_array, 0.0)
        graph = Graph(dist_array)
        self.tsp_problem = TSPProblem(graph, coords_array.tolist())
        self._bounds = self._estimate_bounds(dist_array)

    @staticmethod
    def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
        diff = coords[:, None, :] - coords[None, :, :]
        return np.linalg.norm(diff, axis=2)

    @staticmethod
    def _estimate_bounds(distances: np.ndarray) -> Dict[str, float]:
        if distances.size == 0:
            return {"lower_bound": 0.0, "upper_bound": 1.0}
        upper = float(np.max(distances)) * max(1, distances.shape[0])
        mask = distances > 0.0
        if np.any(mask):
            lower = float(np.min(distances[mask])) * max(1, distances.shape[0])
        else:
            lower = 0.0
        if upper <= lower:
            upper = lower + max(1.0, float(np.mean(distances)) * distances.shape[0])
        return {"lower_bound": max(0.0, lower), "upper_bound": max(upper, lower + 1.0)}

    def _sample_num_cities(self, rng: np.random.Generator) -> int:
        if self._num_cities_range is not None:
            low, high = self._num_cities_range
            return int(rng.integers(low, high + 1))
        assert self._num_cities_fixed is not None
        return int(self._num_cities_fixed)
