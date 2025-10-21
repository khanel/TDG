"""
Problem definitions and utilities for Bees Algorithm (BA).

This module provides base problem classes and utilities that can be used
with the Bees Algorithm implementation.
"""

from Core.problem import ProblemInterface, Solution
import numpy as np

class BAProblem(ProblemInterface):
    """
    Base problem class for Bees Algorithm.

    This provides a template for implementing problems that work with BA.
    """

    def __init__(self, **kwargs):
        """Initialize the problem with BA-specific parameters if needed."""
        super().__init__()
        self._config = kwargs

    def evaluate(self, solution):
        """Evaluate a solution. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement evaluate()")

    def get_initial_solution(self):
        """Generate initial solution. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_initial_solution()")

    def get_problem_info(self):
        """Return problem information. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_problem_info()")


class ContinuousOptimizationProblem(BAProblem):
    """
    Example continuous optimization problem for BA.

    This implements a simple benchmark function (Sphere function).
    """

    def __init__(self, dimension=10, lower_bound=-5.0, upper_bound=5.0, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, solution):
        """Evaluate using Sphere function: f(x) = sum(x_i^2)"""
        x = np.array(solution.representation)
        return np.sum(x ** 2)

    def get_initial_solution(self):
        """Generate random initial solution within bounds."""
        representation = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
        return Solution(representation, self)

    def get_problem_info(self):
        """Return problem information."""
        return {
            'dimension': self.dimension,
            'lower_bounds': [self.lower_bound] * self.dimension,
            'upper_bounds': [self.upper_bound] * self.dimension,
            'problem_type': 'continuous'
        }


class DiscreteOptimizationProblem(BAProblem):
    """
    Example discrete optimization problem for BA.

    This implements a simple discrete problem (can be adapted for TSP, etc.).
    """

    def __init__(self, dimension=10, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension

    def evaluate(self, solution):
        """Simple discrete evaluation (can be overridden for specific problems)."""
        # Example: minimize sum of values
        return sum(solution.representation)

    def get_initial_solution(self):
        """Generate random initial discrete solution."""
        representation = np.random.randint(0, 10, self.dimension).tolist()
        return Solution(representation, self)

    def get_problem_info(self):
        """Return problem information."""
        return {
            'dimension': self.dimension,
            'problem_type': 'discrete'
        }


# Utility functions for BA
def create_ba_parameters(population_size, problem_type='continuous'):
    """
    Create default BA parameters based on population size and problem type.

    Args:
        population_size: Total number of scout bees
        problem_type: 'continuous' or 'discrete'

    Returns:
        Dictionary with BA parameters
    """
    if problem_type == 'continuous':
        return {
            'm': max(5, population_size // 5),  # Selected sites
            'e': max(2, population_size // 10),  # Elite sites
            'nep': 10,  # Bees per elite site
            'nsp': 5,   # Bees per non-elite site
            'ngh': 0.1  # Neighborhood radius
        }
    else:  # discrete
        return {
            'm': max(5, population_size // 5),
            'e': max(2, population_size // 10),
            'nep': 8,
            'nsp': 4,
            'ngh': 3.0  # For discrete, ngh represents number of swaps
        }