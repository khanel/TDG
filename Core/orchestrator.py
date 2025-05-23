"""
HybridSearchOrchestrator: Modular orchestrator for hybrid/metaheuristic strategies.
Coordinates multiple SearchAlgorithm instances and enables flexible hybridization.
"""
from typing import Dict, Callable, List, Optional
from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm

class HybridSearchOrchestrator:
    def __init__(self, problem: ProblemInterface, algorithms: Dict[str, SearchAlgorithm], strategy: Callable[[int, Dict[str, SearchAlgorithm]], str], max_iterations: int = 100):
        """
        Args:
            problem: The problem instance (implements ProblemInterface)
            algorithms: Dict mapping algorithm names to SearchAlgorithm instances
            strategy: Function deciding which algorithm to run at each iteration
            max_iterations: Total number of iterations to run
        """
        self.problem = problem
        self.algorithms = algorithms
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.shared_population: Optional[List[Solution]] = None
        self.best_solution: Optional[Solution] = None
        self.iteration = 0
        self.history = []

    def initialize(self, population_size: int):
        """Initialize all algorithms with a shared population."""
        self.shared_population = [self.problem.get_initial_solution() for _ in range(population_size)]
        for alg in self.algorithms.values():
            alg.population = [sol.copy() for sol in self.shared_population]
            alg._update_best_solution()
        self._update_best_solution()

    def step(self):
        """Run one iteration of the orchestrator using the selected algorithm."""
        alg_name = self.strategy(self.iteration, self.algorithms)
        algorithm = self.algorithms[alg_name]
        # Sync population
        algorithm.population = [sol.copy() for sol in self.shared_population]
        algorithm.step()
        # Update shared population
        self.shared_population = [sol.copy() for sol in algorithm.population]
        self._update_best_solution()
        self.history.append((self.iteration, alg_name, self.best_solution.fitness if self.best_solution else None))
        self.iteration += 1

    def run(self):
        """Run the orchestrator for max_iterations."""
        for _ in range(self.max_iterations):
            self.step()
        
        # Ensure the best solution has its fitness calculated
        if self.best_solution and self.best_solution.fitness is None:
            self.best_solution.evaluate()
            
        return self.best_solution

    def _update_best_solution(self):
        """Update the overall best solution found so far."""
        if self.shared_population:
            # Ensure all solutions have fitness calculated
            for sol in self.shared_population:
                if sol.fitness is None:
                    sol.evaluate()
                    
            current_best = min(self.shared_population, default=None)
            if current_best and (self.best_solution is None or current_best < self.best_solution):
                self.best_solution = current_best.copy()
                
            # Ensure best solution has fitness calculated
            if self.best_solution and self.best_solution.fitness is None:
                self.best_solution.evaluate()

    def get_history(self):
        return self.history

# Example strategy: round-robin

def round_robin_strategy(iteration: int, algorithms: Dict[str, SearchAlgorithm]) -> str:
    alg_names = list(algorithms.keys())
    return alg_names[iteration % len(alg_names)]
