import abc
from typing import Any, Dict, Optional
import numpy as np # Import numpy for potential array comparison

class Solution:
    """Represents a potential solution to the optimization problem."""
    def __init__(self, representation: Any, problem: 'ProblemInterface'):
        self.representation = representation
        self.problem = problem
        self.fitness: Optional[float] = None

    def evaluate(self):
        """Calculates and stores the fitness of this solution."""
        if self.fitness is None:
            self.fitness = self.problem.evaluate(self)
        return self.fitness

    def __lt__(self, other: 'Solution') -> bool:
        """Allows comparison based on fitness (assuming minimization)."""
        if self.fitness is None or other.fitness is None:
            return False # Cannot compare if fitness is unknown
        return self.fitness < other.fitness

    def __eq__(self, other: object) -> bool:
        """Checks if two solutions are equal based on representation."""
        if not isinstance(other, Solution):
            return NotImplemented
        # Use numpy array comparison if applicable, otherwise standard equality
        if isinstance(self.representation, np.ndarray) and isinstance(other.representation, np.ndarray):
            return np.array_equal(self.representation, other.representation)
        return self.representation == other.representation

    def __gt__(self, other: 'Solution') -> bool:
        """Allows comparison based on fitness (assuming minimization)."""
        if self.fitness is None or other.fitness is None:
            return False # Cannot compare if fitness is unknown
        return self.fitness > other.fitness

    def __str__(self) -> str:
        return f"Solution({self.representation}, Fitness: {self.fitness})"

class ProblemInterface(abc.ABC):
    """
    Abstract base class defining the interface for an optimization problem.
    """

    @abc.abstractmethod
    def evaluate(self, solution: Solution) -> float:
        """
        Evaluates the fitness of a given solution. Lower values are better.

        Args:
            solution: The Solution object to evaluate.

        Returns:
            The fitness value (float).
        """
        pass

    @abc.abstractmethod
    def get_initial_solution(self) -> Solution:
        """
        Generates a single, potentially random, valid initial solution.

        Returns:
            A Solution object representing an initial state.
        """
        pass

    @abc.abstractmethod
    def get_problem_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing essential information about the problem.
        Examples: 'dimension', 'lower_bounds', 'upper_bounds', 'problem_type' ('discrete'/'continuous').

        Returns:
            A dictionary with problem-specific details.
        """
        pass

    def get_initial_population(self, population_size: int) -> list[Solution]:
        """
        Generates an initial population of solutions.
        Can be overridden by subclasses for more sophisticated initialization.

        Args:
            population_size: The number of solutions to generate.

        Returns:
            A list of Solution objects.
        """
        return [self.get_initial_solution() for _ in range(population_size)]

    # Optional: Add methods for logging or visualization if common across problems
    # def log_statistics(self, population: list[Solution], iteration: int): pass
    # def plot_progress(self): pass
    # def plot_solution(self, solution: Solution): pass
