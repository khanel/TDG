"""
Knapsack problem adapter for RLOrchestrator.
Wraps Knapsack/knapsack.py to implement Core.problem.ProblemInterface.
"""

from typing import Dict, Any, List
from Knapsack.knapsack import KnapsackProblem
from Core.problem import Solution, ProblemInterface


class KnapsackAdapter(ProblemInterface):
    """Adapter for Knapsack problems."""

    def __init__(self, knapsack_problem: KnapsackProblem):
        self.knapsack_problem = knapsack_problem

    def evaluate(self, solution: Solution) -> float:
        return self.knapsack_problem.evaluate(solution)

    def get_initial_solution(self) -> Solution:
        return self.knapsack_problem.get_initial_solution()

    def get_problem_info(self) -> Dict[str, Any]:
        return self.knapsack_problem.get_problem_info()

    def get_bounds(self) -> Dict[str, float]:
        # Use problem's capacity as upper bound
        return {"lower_bound": 0.0, "upper_bound": float(self.knapsack_problem.capacity)}

    def get_initial_population(self, size: int) -> List[Solution]:
        return [self.get_initial_solution() for _ in range(size)]
