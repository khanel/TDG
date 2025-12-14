"""
Problem helpers tailored for the Artificial Bee Colony (ABC) algorithm.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from Core.problem import ProblemInterface, Solution


class ABCProblem(ProblemInterface):
    """Convenience base class for problems optimized with ABC."""

    def __init__(self, **kwargs):
        super().__init__()
        self._config = dict(kwargs)

    def evaluate(self, solution: Solution) -> float:
        raise NotImplementedError("Subclasses must implement evaluate().")

    def get_initial_solution(self) -> Solution:
        raise NotImplementedError("Subclasses must implement get_initial_solution().")

    def get_problem_info(self) -> Dict[str, object]:
        return dict(self._config)


class ContinuousOptimizationProblem(ABCProblem):
    """Simple continuous benchmark (Sphere function)."""

    def __init__(self, dimension: int = 10, lower_bound: float = -5.0, upper_bound: float = 5.0):
        super().__init__(dimension=dimension, lower_bounds=[lower_bound] * dimension, upper_bounds=[upper_bound] * dimension, problem_type="continuous")
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, solution: Solution) -> float:
        vector = np.asarray(solution.representation, dtype=float)
        return float(np.sum(vector ** 2))

    def get_initial_solution(self) -> Solution:
        vector = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dimension)
        return Solution(vector, self)


class BinaryOptimizationProblem(ABCProblem):
    """Example binary decision problem where the objective is to minimize ones count."""

    def __init__(self, dimension: int = 32):
        super().__init__(dimension=dimension, problem_type="binary")
        self.dimension = dimension

    def evaluate(self, solution: Solution) -> float:
        bits = np.asarray(solution.representation, dtype=int)
        if bits.shape[0] != self.dimension:
            raise ValueError("Representation length mismatch.")
        return float(np.sum(bits))

    def get_initial_solution(self) -> Solution:
        vector = np.random.randint(0, 2, size=self.dimension)
        return Solution(vector.tolist(), self)


class PermutationProblem(ABCProblem):
    """Permutation benchmark that minimizes the tour length defined by a distance matrix."""

    def __init__(self, distance_matrix: np.ndarray):
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square.")
        n = int(distance_matrix.shape[0])
        super().__init__(dimension=n, problem_type="permutation")
        self.distance_matrix = np.asarray(distance_matrix, dtype=float)
        self.n = n

    def evaluate(self, solution: Solution) -> float:
        tour = list(solution.representation)
        if len(tour) != self.n:
            raise ValueError("Tour length mismatch.")
        cost = 0.0
        for i in range(self.n):
            a = tour[i % self.n]
            b = tour[(i + 1) % self.n]
            cost += self.distance_matrix[a % self.n, b % self.n]
        return cost

    def get_initial_solution(self) -> Solution:
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        return Solution(perm.tolist(), self)


def create_abc_parameters(
    population_size: int,
    dimension: int,
    problem_type: str = "continuous",
    *,
    onlooker_ratio: float = 1.0,
) -> Dict[str, object]:
    """
    Provide a reasonable default parameter dictionary for ABC experiments.
    """
    dim = max(1, int(dimension))
    onlookers = max(1, int(onlooker_ratio * population_size))
    if problem_type in {"permutation", "discrete", "binary"}:
        limit = int(1.5 * population_size * dim)
        perturb_scale = 1.0
    else:
        limit = int(population_size * dim)
        perturb_scale = 0.5
    return {
        "onlooker_count": onlookers,
        "limit": limit,
        "perturbation_scale": perturb_scale,
    }
