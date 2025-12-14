"""Knapsack problem adapter with optional per-episode randomization."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from problems.Knapsack.knapsack import KnapsackProblem
from Core.problem import Solution, ProblemInterface


@dataclass
class KnapsackRandomSpec:
    n_items: int
    value_range: Tuple[float, float]
    weight_range: Tuple[float, float]
    capacity_ratio: float = 0.5
    seed: Optional[int] = None


def _generate_random_problem(spec: KnapsackRandomSpec) -> KnapsackProblem:
    rng = np.random.default_rng(spec.seed)
    values = rng.uniform(spec.value_range[0], spec.value_range[1], size=spec.n_items)
    weights = rng.uniform(spec.weight_range[0], spec.weight_range[1], size=spec.n_items)
    capacity = spec.capacity_ratio * float(np.sum(weights))
    return KnapsackProblem(values, weights, capacity, seed=spec.seed)


class KnapsackAdapter(ProblemInterface):
    """Adapter around :class:`KnapsackProblem` with regeneration hooks."""

    def __init__(
        self,
        knapsack_problem: Optional[KnapsackProblem] = None,
        *,
        values: Optional[Iterable[float]] = None,
        weights: Optional[Iterable[float]] = None,
        capacity: Optional[float] = None,
        n_items: int = 50,
        value_range: Tuple[float, float] = (1.0, 100.0),
        weight_range: Tuple[float, float] = (1.0, 50.0),
        capacity_ratio: float = 0.5,
        seed: Optional[int] = 42,
    ):
        self._randomizable = False
        self._rng = np.random.default_rng(seed)
        self._n_items_range: Optional[Tuple[int, int]] = None
        self._n_items_fixed: Optional[int] = None
        self._value_range = tuple(value_range)
        self._weight_range = tuple(weight_range)
        self._capacity_ratio = float(capacity_ratio)

        if isinstance(n_items, (tuple, list)) and len(n_items) == 2:
            lo = int(float(n_items[0]))
            hi = int(float(n_items[1]))
            lo, hi = sorted((lo, hi))
            self._n_items_range = (max(1, lo), max(1, hi))
        else:
            self._n_items_fixed = max(1, int(n_items))

        if knapsack_problem is not None:
            self.knapsack_problem = knapsack_problem
        elif values is not None and weights is not None and capacity is not None:
            self.knapsack_problem = KnapsackProblem(values, weights, capacity, seed=seed)
        else:
            self._randomizable = True
            sample_n = self._sample_num_items(self._rng)
            spec = KnapsackRandomSpec(
                n_items=sample_n,
                value_range=value_range,
                weight_range=weight_range,
                capacity_ratio=float(capacity_ratio),
                seed=seed,
            )
            self.knapsack_problem = _generate_random_problem(spec)
            self._value_range = tuple(value_range)
            self._weight_range = tuple(weight_range)
            self._capacity_ratio = float(capacity_ratio)

        self._update_bounds()

    def evaluate(self, solution: Solution) -> float:
        mask = np.asarray(solution.representation, dtype=float)
        values = self._to_numpy(self.knapsack_problem.values)
        weights = self._to_numpy(self.knapsack_problem.weights)
        capacity = float(self.knapsack_problem.capacity)
        penalty_factor = float(self.knapsack_problem.penalty_factor)
        total_value = float(np.dot(values, mask))
        total_weight = float(np.dot(weights, mask))
        overflow = max(0.0, total_weight - capacity)
        fitness = -total_value + penalty_factor * overflow
        solution.fitness = fitness
        solution._cached_total_value = total_value
        solution._cached_total_weight = total_weight
        return fitness

    def _to_numpy(self, arr) -> np.ndarray:
        if hasattr(arr, "get"):
            return np.asarray(arr.get(), dtype=float)
        return np.asarray(arr, dtype=float)

    def _random_mask(self) -> np.ndarray:
        rng = self._rng
        n = int(self._to_numpy(self.knapsack_problem.values).size)
        mask = rng.binomial(1, 0.5, size=n).astype(float)
        return self._repair_mask(mask)

    def _repair_mask(self, mask: np.ndarray) -> np.ndarray:
        weights = self._to_numpy(self.knapsack_problem.weights)
        capacity = float(self.knapsack_problem.capacity)
        mask = mask.astype(float, copy=True)
        total_weight = float(np.dot(weights, mask))
        if total_weight <= capacity:
            return mask
        active = np.where(mask > 0.5)[0]
        rng = self._rng
        rng.shuffle(active)
        for idx in active:
            mask[idx] = 0.0
            total_weight -= weights[idx]
            if total_weight <= capacity:
                break
        return mask

    def get_initial_solution(self) -> Solution:
        mask = self._random_mask()
        sol = Solution(mask.astype(int).tolist(), self)
        sol.evaluate()
        return sol

    def get_initial_population(self, size: int) -> List[Solution]:
        return [self.get_initial_solution() for _ in range(max(1, int(size)))]

    def get_problem_info(self) -> Dict[str, Any]:
        values = self._to_numpy(self.knapsack_problem.values)
        weights = self._to_numpy(self.knapsack_problem.weights)
        return {
            "dimension": int(values.size),
            "problem_type": "binary",
            "capacity": float(self.knapsack_problem.capacity),
            "values": values.copy(),
            "weights": weights.copy(),
            "lower_bounds": np.zeros(values.size, dtype=float),
            "upper_bounds": np.ones(values.size, dtype=float),
            "penalty_factor": float(self.knapsack_problem.penalty_factor),
        }

    def get_bounds(self) -> Dict[str, float]:
        return dict(self._bounds)

    def regenerate_instance(self) -> bool:
        if not self._randomizable:
            return False
        seed = int(self._rng.integers(0, 2**31))
        value_range = getattr(self, "_value_range", (1.0, 100.0))
        weight_range = getattr(self, "_weight_range", (1.0, 50.0))
        capacity_ratio = float(getattr(self, "_capacity_ratio", 0.5))
        sample_n = self._sample_num_items(self._rng)
        spec = KnapsackRandomSpec(
            n_items=sample_n,
            value_range=value_range,
            weight_range=weight_range,
            capacity_ratio=capacity_ratio,
            seed=seed,
        )
        self.knapsack_problem = _generate_random_problem(spec)
        self._update_bounds()
        return True

    def _update_bounds(self) -> None:
        total_value = getattr(self.knapsack_problem, "_total_value", float(np.sum(self._to_numpy(self.knapsack_problem.values))))
        self._bounds = {"lower_bound": -total_value, "upper_bound": 0.0}

    def repair_mask(self, mask: Iterable[int]) -> np.ndarray:
        mask_arr = np.asarray(list(mask), dtype=float)
        return self._repair_mask(mask_arr)

    def _sample_num_items(self, rng: np.random.Generator) -> int:
        if self._n_items_range is not None:
            low, high = self._n_items_range
            return int(rng.integers(low, high + 1))
        assert self._n_items_fixed is not None
        return int(self._n_items_fixed)
