# rastrigin_problem.py
import math
import numpy as np
from typing import Any, Dict
from Core.problem import ProblemInterface, Solution

class Rastrigin(ProblemInterface):
    def __init__(self, dim: int = 10, lower: float = -5.12, upper: float = 5.12):
        self.dim = dim
        self.lower = lower
        self.upper = upper

    def evaluate(self, solution: Solution) -> float:
        x = np.asarray(solution.representation, dtype=float)
        A = 10.0
        return A * self.dim + float(np.sum(x * x - A * np.cos(2.0 * math.pi * x)))

    def get_initial_solution(self) -> Solution:
        x0 = np.random.uniform(self.lower, self.upper, size=self.dim)
        return Solution(x0, self)

    def get_problem_info(self) -> Dict[str, Any]:
        return {
            "dimension": self.dim,
            "lower_bounds": np.full(self.dim, self.lower, dtype=float),
            "upper_bounds": np.full(self.dim, self.upper, dtype=float),
            "problem_type": "continuous"
        }
