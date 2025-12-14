"""
Problem definitions and utilities for Gradient-Based Optimizer (GBO).

This module provides base problem classes and utilities that can be used
with the GBO implementation.
"""

from Core.problem import ProblemInterface, Solution
import numpy as np

class GBOProblem(ProblemInterface):
    """
    Base problem class for Gradient-Based Optimizer.

    This provides a template for implementing problems that work with GBO.
    """

    def __init__(self, **kwargs):
        """Initialize the problem with GBO-specific parameters if needed."""
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


class ContinuousOptimizationProblem(GBOProblem):
    """
    Example continuous optimization problem for GBO.

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


class AckleyProblem(GBOProblem):
    """
    Ackley function - a multimodal benchmark problem for GBO.

    f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
    Global minimum at x = 0, f(x) = 0
    """

    def __init__(self, dimension=10, lower_bound=-5.0, upper_bound=5.0, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, solution):
        """Evaluate using Ackley function."""
        x = np.array(solution.representation)
        d = len(x)

        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
        term3 = 20 + np.e

        return term1 + term2 + term3

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


class GriewankProblem(GBOProblem):
    """
    Griewank function - a multimodal benchmark problem.

    f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i)))
    Global minimum at x = 0, f(x) = 0
    """

    def __init__(self, dimension=10, lower_bound=-600.0, upper_bound=600.0, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, solution):
        """Evaluate using Griewank function."""
        x = np.array(solution.representation)

        sum_term = np.sum(x**2) / 4000
        prod_term = 1.0
        for i in range(len(x)):
            prod_term *= np.cos(x[i] / np.sqrt(i + 1))

        return 1 + sum_term - prod_term

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


class LevyProblem(GBOProblem):
    """
    Levy function - a multimodal benchmark problem.

    f(x) = sin^2(π*w1) + sum((w_i-1)^2 * (1 + 10*sin^2(π*w_i+1))) + (w_d-1)^2
    where w_i = 1 + (x_i - 1)/4
    Global minimum at x = 1, f(x) = 0
    """

    def __init__(self, dimension=10, lower_bound=-10.0, upper_bound=10.0, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, solution):
        """Evaluate using Levy function."""
        x = np.array(solution.representation)
        d = len(x)

        # Transform variables
        w = 1 + (x - 1) / 4

        # First term
        term1 = np.sin(np.pi * w[0])**2

        # Sum term
        sum_term = 0
        for i in range(d - 1):
            sum_term += (w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2)

        # Last term
        term3 = (w[d-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[d-1])**2)

        return term1 + sum_term + term3

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


# Utility functions for GBO
def create_gbo_parameters(population_size, problem_type='continuous', exploration_focus='balanced'):
    """
    Create default GBO parameters based on population size and problem type.

    Args:
        population_size: Number of agents in population
        problem_type: 'continuous' or 'discrete'
        exploration_focus: 'exploration', 'balanced', or 'exploitation'

    Returns:
        Dictionary with GBO parameters
    """
    if exploration_focus == 'exploration':
        return {
            'alpha': 1.5,
            'beta': 1.5,
            'leo_prob': 0.2,
            'step_size': 1.2,
            'exploration_boost': 1.8
        }
    elif exploration_focus == 'exploitation':
        return {
            'alpha': 0.8,
            'beta': 0.8,
            'leo_prob': 0.05,
            'step_size': 0.8,
            'exploration_boost': 1.0
        }
    else:  # balanced
        return {
            'alpha': 1.0,
            'beta': 1.0,
            'leo_prob': 0.1,
            'step_size': 1.0,
            'exploration_boost': 1.3
        }


def adaptive_gbo_parameters(max_iterations, current_iteration):
    """
    Create adaptive GBO parameters that change over time.

    Args:
        max_iterations: Maximum number of iterations
        current_iteration: Current iteration number

    Returns:
        Dictionary with adaptive parameters
    """
    progress = current_iteration / max_iterations

    if progress < 0.3:
        # Early phase: high exploration
        return {
            'alpha': 1.5,
            'beta': 1.5,
            'leo_prob': 0.2,
            'step_size': 1.2
        }
    elif progress < 0.7:
        # Mid phase: balanced
        return {
            'alpha': 1.0,
            'beta': 1.0,
            'leo_prob': 0.1,
            'step_size': 1.0
        }
    else:
        # Late phase: exploitation
        return {
            'alpha': 0.8,
            'beta': 0.8,
            'leo_prob': 0.05,
            'step_size': 0.6
        }


def create_problem_specific_gbo_parameters(problem_name):
    """
    Create GBO parameters optimized for specific benchmark problems.

    Args:
        problem_name: Name of the benchmark problem

    Returns:
        Dictionary with optimized parameters
    """
    if problem_name.lower() == 'sphere':
        return {
            'alpha': 1.2,
            'beta': 1.2,
            'leo_prob': 0.15,
            'step_size': 0.8,
            'exploration_boost': 1.4
        }
    elif problem_name.lower() == 'rastrigin':
        return {
            'alpha': 1.8,
            'beta': 1.8,
            'leo_prob': 0.25,
            'step_size': 1.5,
            'exploration_boost': 2.0
        }
    elif problem_name.lower() == 'ackley':
        return {
            'alpha': 1.5,
            'beta': 1.5,
            'leo_prob': 0.2,
            'step_size': 1.2,
            'exploration_boost': 1.8
        }
    else:  # default
        return {
            'alpha': 1.0,
            'beta': 1.0,
            'leo_prob': 0.1,
            'step_size': 1.0,
            'exploration_boost': 1.3
        }