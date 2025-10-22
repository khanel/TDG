import abc
from itertools import count
from typing import Any, Dict, Optional
import numpy as np # Import numpy for potential array comparison

class Solution:
    """Represents a potential solution to the optimization problem."""
    _id_counter = count()

    def __init__(self, representation: Any, problem: 'ProblemInterface', *, solution_id: Optional[int] = None):
        self.representation = representation
        self.problem = problem
        self.fitness: Optional[float] = None
        # Assign a stable identifier so downstream components can track individuals cheaply.
        self.id: int = int(next(self._id_counter) if solution_id is None else solution_id)
        # Cache for incremental updates (to be set by problem.evaluate if supported)
        self._cached_total_value: Optional[float] = None
        self._cached_total_weight: Optional[float] = None

    def evaluate(self):
        """Calculates and stores the fitness of this solution."""
        if self.fitness is None:
            self.fitness = self.problem.evaluate(self)
        return self.fitness

    def copy(self, *, preserve_id: bool = True):
        """Creates a deep copy of this solution.

        Args:
            preserve_id: When True (default), the cloned solution keeps the same `id`.
                Set to False if the copy represents a genuinely new individual.
        """
        import copy
        new_id = self.id if preserve_id else None
        new_solution = Solution(copy.deepcopy(self.representation), self.problem, solution_id=new_id)
        new_solution.fitness = self.fitness
        new_solution._cached_total_value = self._cached_total_value
        new_solution._cached_total_weight = self._cached_total_weight
        return new_solution

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
