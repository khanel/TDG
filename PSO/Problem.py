"""
Problem definitions and utilities for Particle Swarm Optimization (PSO).

This module provides base problem classes and utilities that can be used
with the PSO implementation.
"""

from Core.problem import ProblemInterface, Solution
import numpy as np

class PSOProblem(ProblemInterface):
    """
    Base problem class for Particle Swarm Optimization.

    This provides a template for implementing problems that work with PSO.
    """

    def __init__(self, **kwargs):
        """Initialize the problem with PSO-specific parameters if needed."""
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


class ContinuousOptimizationProblem(PSOProblem):
    """
    Example continuous optimization problem for PSO.

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


class RastriginProblem(PSOProblem):
    """
    Rastrigin function - a multimodal benchmark problem for PSO.

    f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    Global minimum at x = 0, f(x) = 0
    """

    def __init__(self, dimension=10, lower_bound=-5.12, upper_bound=5.12, A=10, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.A = A

    def evaluate(self, solution):
        """Evaluate using Rastrigin function."""
        x = np.array(solution.representation)
        return self.A * self.dimension + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))

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


class RosenbrockProblem(PSOProblem):
    """
    Rosenbrock function - a classic optimization benchmark.

    f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    Global minimum at (x,y) = (1,1), f(x,y) = 0
    """

    def __init__(self, dimension=2, lower_bound=-2.0, upper_bound=2.0, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, solution):
        """Evaluate using Rosenbrock function."""
        x = np.array(solution.representation)
        result = 0
        for i in range(len(x) - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return result

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


# Utility functions for PSO
def create_pso_parameters(population_size, problem_type='continuous', use_constriction=False):
    """
    Create default PSO parameters based on population size and problem type.

    Args:
        population_size: Number of particles in swarm
        problem_type: 'continuous' or 'discrete'
        use_constriction: Whether to use constriction factor

    Returns:
        Dictionary with PSO parameters
    """
    if use_constriction:
        # Constriction factor parameters (Clerc-Kennedy)
        return {
            'omega': None,  # Not used with constriction
            'c1': 2.05,
            'c2': 2.05,
            'use_constriction': True,
            'vmax_factor': 0.2
        }
    else:
        # Inertia weight parameters
        return {
            'omega': 0.7,
            'c1': 1.5,
            'c2': 1.5,
            'use_constriction': False,
            'vmax_factor': 0.2
        }


def linear_inertia_schedule(initial_omega=0.9, final_omega=0.4, max_iterations=100):
    """
    Create a linear inertia weight schedule.

    Args:
        initial_omega: Starting inertia weight
        final_omega: Final inertia weight
        max_iterations: Maximum number of iterations

    Returns:
        Function that returns omega for a given iteration
    """
    def get_omega(iteration):
        if iteration >= max_iterations:
            return final_omega
        return initial_omega - (initial_omega - final_omega) * (iteration / max_iterations)

    return get_omega


def create_adaptive_pso_parameters(population_size, problem_complexity='moderate'):
    """
    Create adaptive PSO parameters based on problem complexity.

    Args:
        population_size: Number of particles
        problem_complexity: 'simple', 'moderate', or 'complex'

    Returns:
        Dictionary with adaptive PSO parameters
    """
    if problem_complexity == 'simple':
        return {
            'omega': 0.8,
            'c1': 1.2,
            'c2': 1.2,
            'use_constriction': False,
            'vmax_factor': 0.1,
            'exploration_boost': 1.2
        }
    elif problem_complexity == 'complex':
        return {
            'omega': 0.6,
            'c1': 2.0,
            'c2': 2.0,
            'use_constriction': True,
            'vmax_factor': 0.3,
            'exploration_boost': 1.8
        }
    else:  # moderate
        return {
            'omega': 0.7,
            'c1': 1.5,
            'c2': 1.5,
            'use_constriction': False,
            'vmax_factor': 0.2,
            'exploration_boost': 1.5
        }