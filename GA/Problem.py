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
    """
    def __init__(self, representation, problem):
        self.representation = representation
        self.problem = problem
        self.fitness = None

    def evaluate(self):
        self.fitness = self.problem.calculate_fitness(self.representation)



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
