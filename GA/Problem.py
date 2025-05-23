from abc import ABC, abstractmethod

# Problem Interface
class ProblemInterface(ABC):
    @abstractmethod
    def generate_initial_population(self, size):
        pass

    @abstractmethod
    def calculate_fitness(self, individual):
        pass

    @abstractmethod
    def validate_individual(self, individual):
        pass

    @abstractmethod
    def get_individual_size(self):
        pass


# Solution Class
class Solution:
    """
    Class representing a solution to a problem.

    Attributes
    ----------
    representation: any
        The representation of the solution.
    problem: ProblemInterface
        The problem that this solution is for.
    fitness: float
        The fitness of this solution.

    Methods
    -------
    evaluate()
        Evaluates this solution and sets its fitness.
    copy()
        Creates a deep copy of this solution.
    """
    def __init__(self, representation, problem):
        self.representation = representation
        self.problem = problem
        self.fitness = None

    def evaluate(self):
        # Use problem.evaluate instead of problem.calculate_fitness
        if hasattr(self.problem, 'evaluate'):
            self.fitness = self.problem.evaluate(self)
        elif hasattr(self.problem, 'calculate_fitness'):
            self.fitness = self.problem.calculate_fitness(self.representation)
        return self.fitness
        
    def copy(self):
        import copy
        new_solution = Solution(copy.deepcopy(self.representation), self.problem)
        new_solution.fitness = self.fitness
        return new_solution
        
    def __lt__(self, other):
        """Allows comparison based on fitness (assuming minimization)."""
        if self.fitness is None or other.fitness is None:
            return False  # Cannot compare if fitness is unknown
        return self.fitness < other.fitness

    def __gt__(self, other):
        """Allows comparison based on fitness (assuming minimization)."""
        if self.fitness is None or other.fitness is None:
            return False  # Cannot compare if fitness is unknown
        return self.fitness > other.fitness

    def __eq__(self, other):
        """Checks if two solutions are equal based on representation."""
        if not isinstance(other, Solution):
            return NotImplemented
        # Try numpy array comparison if applicable
        try:
            import numpy as np
            if isinstance(self.representation, np.ndarray) and isinstance(other.representation, np.ndarray):
                return np.array_equal(self.representation, other.representation)
        except ImportError:
            pass
        return self.representation == other.representation



class GeneticOperator(ABC):
    """
    Abstract base class for all genetic operators.

    Attributes
    ----------
    None

    Methods
    -------
    select(population)
        Selects two parents from the given population.
    crossover(parent1, parent2)
        Creates a child from the two given parents.
    mutate(individual)
        Mutates the given individual.

    Notes
    -----
    All genetic operators must implement these three methods.
    """
    @abstractmethod
    def select(self, population): pass

    @abstractmethod
    def crossover(self, parent1, parent2): pass

    @abstractmethod
    def mutate(self, individual): pass
