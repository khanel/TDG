
"""
This file contains the self-contained implementation of the NK-Landscape
problem, including the core problem logic and the adapter required to
interface with the reinforcement learning environment.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# We need the base classes from core.py
from temp.core.base import ProblemInterface, Solution


class NKLProblem:
    """
    Generates and evaluates NK-Landscape problems using a fully vectorized approach.
    """

    def __init__(self, n: int, k: int, seed: int = None):
        if not (0 <= k < n):
            raise ValueError("K must be in the range [0, N-1]")

        self.n = n
        self.k = k
        self.rng = np.random.default_rng(seed)

        self._connections = self._generate_connections()
        self._all_indices = np.zeros((n, k + 1), dtype=int)
        self._all_indices[:, 0] = np.arange(n)
        self._all_indices[:, 1:] = self._connections
        self._powers_of_2 = 1 << np.arange(k, -1, -1)
        num_entries = 2 ** (k + 1)
        self.tables = self.rng.random((n, num_entries))
        self._row_indices = np.arange(n)

    def _generate_connections(self) -> np.ndarray:
        connections = np.zeros((self.n, self.k), dtype=int)
        for i in range(self.n):
            possible_neighbors = list(range(self.n))
            possible_neighbors.remove(i)
            connections[i] = self.rng.choice(possible_neighbors, self.k, replace=False)
        return connections

    def evaluate(self, solution: np.ndarray) -> float:
        if solution.ndim != 1:
            solution = solution.flatten()
        
        patterns = solution[self._all_indices]
        indices = patterns.dot(self._powers_of_2)
        contributions = self.tables[self._row_indices, indices]
        total_fitness = np.sum(contributions)

        return -(total_fitness / self.n)
    
    def batch_evaluate(self, solutions: np.ndarray) -> np.ndarray:
        """
        Highly optimized vectorized batch evaluation for multiple solutions.
        Args:
            solutions: Array of shape (batch_size, n) containing multiple solutions
        Returns:
            Array of shape (batch_size,) containing fitness values
        """
        if solutions.ndim == 1:
            return np.array([self.evaluate(solutions)])
        
        batch_size = solutions.shape[0]
        
        # Pre-allocate arrays for better performance
        patterns = solutions[:, self._all_indices]  # Shape: (batch_size, n, k+1)
        indices = np.dot(patterns, self._powers_of_2)  # Shape: (batch_size, n)
        
        # Optimized vectorized table lookup using advanced indexing
        row_indices = np.arange(self.n)
        contributions = self.tables[row_indices[:, None], indices.T]  # Shape: (n, batch_size)
        
        # Vectorized sum and normalization
        total_fitness = np.sum(contributions, axis=0)  # Shape: (batch_size,)
        
        return -(total_fitness / self.n)


class NKLAdapter(ProblemInterface):
    """Adapter for the NKLProblem."""

    def __init__(self, n_items: int = 100, k_interactions: int = 5, seed: Optional[int] = 42):
        self._rng = np.random.default_rng(seed)
        self._n_items_range = (max(2, n_items), max(2, n_items))
        self._k_interactions_range = (max(0, k_interactions), max(0, k_interactions))
        self.nkl_problem = self._create_problem_instance()
        self._update_bounds()

    def _create_problem_instance(self) -> NKLProblem:
        n = self._rng.integers(self._n_items_range[0], self._n_items_range[1] + 1)
        max_k = n - 1
        k_min = min(max_k, self._k_interactions_range[0])
        k_max = min(max_k, self._k_interactions_range[1])
        k = self._rng.integers(k_min, k_max + 1) if k_min < k_max else k_min
        seed = int(self._rng.integers(0, 2**31))
        return NKLProblem(n=n, k=k, seed=seed)

    def evaluate(self, solution: Solution) -> float:
        rep = np.array(solution.representation, dtype=int)
        n = self.nkl_problem.n
        if rep.shape[0] != n:
            # Simple resize for compatibility
            new_rep = np.zeros(n, dtype=int)
            size = min(n, rep.shape[0])
            new_rep[:size] = rep[:size]
            solution.representation = new_rep
        
        fitness = self.nkl_problem.evaluate(np.array(solution.representation))
        solution.fitness = fitness
        return fitness
    
    def batch_evaluate_solutions(self, solutions: List['Solution']) -> np.ndarray:
        """
        Highly optimized vectorized batch evaluation for multiple Solution objects.
        Args:
            solutions: List of Solution objects
        Returns:
            Array of fitness values
        """
        if not solutions:
            return np.array([])
        
        # Vectorized representation extraction and normalization
        n = self.nkl_problem.n
        
        # Pre-allocate array for better performance
        batch_size = len(solutions)
        representations = np.zeros((batch_size, n), dtype=int)
        
        for i, sol in enumerate(solutions):
            rep = np.asarray(sol.representation, dtype=int)
            if rep.shape[0] != n:
                # Handle size mismatch efficiently
                size = min(n, rep.shape[0])
                representations[i, :size] = rep[:size]
            else:
                representations[i] = rep
        
        # Vectorized batch evaluation
        fitnesses = self.nkl_problem.batch_evaluate(representations)
        
        # Update solution fitnesses in-place
        for sol, fitness in zip(solutions, fitnesses):
            sol.fitness = float(fitness)
        
        return fitnesses

    def get_initial_population(self, population_size: int) -> List['Solution']:
        """
        Generates an initial population of solutions using vectorized evaluation.
        """
        n = self.nkl_problem.n
        
        # Vectorized population generation
        population_representations = self._rng.integers(0, 2, size=(population_size, n), dtype=int)
        
        # Create solution objects
        population = [Solution(representation, self) for representation in population_representations]
        
        # Use vectorized batch evaluation
        self.batch_evaluate_solutions(population)
        
        return population

    def get_initial_solution(self) -> Solution:
        n = self.nkl_problem.n
        mask = self._rng.integers(0, 2, size=n, dtype=int)
        sol = Solution(mask, self)
        sol.evaluate()
        return sol

    def get_problem_info(self) -> Dict[str, Any]:
        n = self.nkl_problem.n
        return {
            "dimension": n,
            "problem_type": "binary",
            "lower_bound": -1.0,
            "upper_bound": 0.0,
        }

    def get_bounds(self) -> Dict[str, float]:
        return {"lower_bound": -1.0, "upper_bound": 0.0}

    def regenerate_instance(self) -> bool:
        self.nkl_problem = self._create_problem_instance()
        self._update_bounds()
        return True

    def _update_bounds(self):
        # Static for NKL
        pass
