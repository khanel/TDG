from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import numpy as np

from Core.problem import ProblemInterface, Solution


class KnapsackProblem(ProblemInterface):
    """0/1 knapsack posed as a minimization task (maximize value, penalize overflow)."""

    def __init__(
        self,
        values: Iterable[float],
        weights: Iterable[float],
        capacity: float,
        *,
        penalty_factor: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        vals = np.asarray(list(values), dtype=float)
        wts = np.asarray(list(weights), dtype=float)
        if vals.shape != wts.shape:
            raise ValueError("values and weights must have matching lengths")
        if np.any(wts <= 0):
            raise ValueError("weights must be strictly positive")
        capacity = float(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.values = vals
        self.weights = wts
        self.capacity = capacity
        density = vals / wts
        base_penalty = float(np.max(vals)) if vals.size else 1.0
        self.penalty_factor = float(penalty_factor) if penalty_factor is not None else max(1.0, 2.0 * base_penalty)
        self._rng = np.random.default_rng(seed)
        self._value_bounds = (0.0, float(np.sum(vals)))
        self._weight_bounds = (0.0, float(np.sum(wts)))
        self._density = density
        self._sorted_density_indices = np.argsort(density)  # Precompute sorted indices by density (ascending)

    # ---- ProblemInterface API ----
    def evaluate(self, solution: Solution) -> float:
        mask = self._to_vector(solution.representation)
        total_value = float(np.dot(self.values, mask))
        total_weight = float(np.dot(self.weights, mask))
        overflow = max(0.0, total_weight - self.capacity)
        penalty = self.penalty_factor * overflow
        fitness = -total_value + penalty
        solution.fitness = fitness
        return fitness

    def get_initial_solution(self) -> Solution:
        mask = self._rng.binomial(1, 0.5, size=self.values.size).astype(float)
        mask = self._repair(mask)
        sol = Solution(mask.astype(int).tolist(), self)
        sol.evaluate()
        return sol

    def get_initial_population(self, population_size: int) -> list[Solution]:
        pop = []
        for _ in range(max(1, int(population_size))):
            mask = self._rng.binomial(1, 0.5, size=self.values.size).astype(float)
            mask = self._repair(mask)
            sol = Solution(mask.astype(int).tolist(), self)
            sol.evaluate()
            pop.append(sol)
        return pop

    def get_problem_info(self) -> dict:
        return {
            "dimension": int(self.values.size),
            "problem_type": "binary",
            "capacity": float(self.capacity),
            "values": self.values.copy(),
            "weights": self.weights.copy(),
            "lower_bounds": np.zeros(self.values.size, dtype=float),
            "upper_bounds": np.ones(self.values.size, dtype=float),
            "penalty_factor": float(self.penalty_factor),
            "value_bounds": tuple(self._value_bounds),
            "weight_bounds": tuple(self._weight_bounds),
        }

    # ---- Domain helpers ----
    def neighbor(self, solution: Solution) -> Solution:
        mask = self._to_vector(solution.representation)
        if mask.size == 0:
            return Solution([], self)
        idx = int(self._rng.integers(0, mask.size))
        mask[idx] = 1.0 - mask[idx]
        if mask[idx] > 0.5 and np.dot(self.weights, mask) > self.capacity:
            # Remove a low-density item until feasible
            mask = self._repair(mask)
        child = Solution(mask.astype(int).tolist(), self)
        return child

    def generate_feasible_mask(self, seed: Optional[int] = None) -> np.ndarray:
        rng = self._rng if seed is None else np.random.default_rng(seed)
        mask = rng.binomial(1, 0.5, size=self.values.size).astype(float)
        return self._repair(mask)

    def repair(self, mask: Iterable) -> np.ndarray:
        vec = self._to_vector(mask)
        return self._repair(vec)

    # ---- internal helpers ----
    def _repair(self, mask: np.ndarray) -> np.ndarray:
        mask = np.clip(mask.astype(float, copy=True), 0.0, 1.0)
        if mask.size == 0:
            return mask
        total_weight = float(np.dot(self.weights, mask))
        if total_weight <= self.capacity:
            return mask
        active = np.where(mask > 0.5)[0]
        if active.size == 0:
            return mask
        order = np.argsort(self._density[active])  # drop weakest items first
        for idx in order:
            j = active[idx]
            mask[j] = 0.0
            total_weight -= float(self.weights[j])
            if total_weight <= self.capacity:
                break
        return mask

    def _to_vector(self, rep: Iterable) -> np.ndarray:
        arr = np.asarray(list(rep), dtype=float)
        if arr.size != self.values.size:
            raise ValueError("representation length mismatch with problem dimension")
        return np.clip(arr, 0.0, 1.0)


@dataclass
class KnapsackSpec:
    n_items: int
    value_range: Tuple[float, float]
    weight_range: Tuple[float, float]
    capacity_ratio: float = 0.5
    seed: Optional[int] = None


def generate_random_knapsack(spec: KnapsackSpec) -> Tuple[KnapsackProblem, dict]:
    rng = np.random.default_rng(spec.seed)
    n = max(1, int(spec.n_items))
    v_low, v_high = spec.value_range
    w_low, w_high = spec.weight_range
    values = rng.uniform(v_low, v_high, size=n)
    weights = rng.uniform(w_low, w_high, size=n)
    # Scale capacity to fraction of total weight (controls tightness)
    capacity = float(spec.capacity_ratio) * float(np.sum(weights))
    problem = KnapsackProblem(values, weights, capacity, seed=spec.seed)

    # Heuristic bounds for normalization
    sorted_indices = np.argsort(-(values / weights))
    best_value = 0.0
    remaining_cap = capacity
    for idx in sorted_indices:
        if weights[idx] <= remaining_cap:
            best_value += values[idx]
            remaining_cap -= weights[idx]
        else:
            frac = remaining_cap / weights[idx]
            best_value += frac * values[idx]
            break
    lower_bound = -float(best_value)
    all_selected_weight = float(np.sum(weights))
    overflow = max(0.0, all_selected_weight - capacity)
    worst_penalty = problem.penalty_factor * overflow
    upper_bound = max(worst_penalty, 0.0)
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + float(np.sum(values))

    bounds = {
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "best_value_bound": float(best_value),
    }
    return problem, bounds
