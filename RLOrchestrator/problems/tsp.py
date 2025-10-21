"""
TSP problem adapter for RLOrchestrator.
Wraps TSP.TSP.TSPProblem to implement Core.problem.ProblemInterface.
"""

from typing import Dict, Any, List
from TSP.TSP import TSPProblem
from Core.problem import Solution, ProblemInterface


class TSPAdapter(ProblemInterface):
    """Adapter for TSP problems."""

    def __init__(self, tsp_problem: TSPProblem):
        self.tsp_problem = tsp_problem

    def evaluate(self, solution: Solution) -> float:
        return self.tsp_problem.evaluate(solution)

    def get_initial_solution(self) -> Solution:
        return self.tsp_problem.get_initial_solution()

    def get_problem_info(self) -> Dict[str, Any]:
        return self.tsp_problem.get_problem_info()

    def get_bounds(self) -> Dict[str, float]:
        # Simplified bounds (use MST approximation)
        n = len(self.tsp_problem.city_coords)
        return {"lower_bound": 0.0, "upper_bound": n * 100.0}  # Rough estimate

    def get_initial_population(self, size: int) -> List[Solution]:
        return [self.get_initial_solution() for _ in range(size)]
