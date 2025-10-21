import abc
import numpy as np

class BaseProblem(abc.ABC):
    @abc.abstractmethod
    def __init__(self, population_size, dim, lower_bound, upper_bound):
        self.population_size = population_size
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def fitness(self, position):
        pass

class GWO_TSPProblem(BaseProblem):
    def __init__(self, graph, city_coords, population_size, dim, lower_bound, upper_bound, fitness_function):
        super().__init__(population_size, dim, lower_bound, upper_bound)
        self.cities_graph = graph
        self.city_coords = city_coords
        self.population = None
        self.fitness_function = fitness_function

    def initialize(self):
        # Implementation for population initialization
        # For now, simply call a method on the problem to set initial state
        if hasattr(self, "initialize"):
            self.initialize()
        else:
            print("Population initialized (default implementation).")

    def fitness(self, position):
        # Implementation for fitness evaluation
        return self.calculate_path_distance(position)

    def evaluate(self, solution):
        # Implementation for solution evaluation (required by Solution class)
        return self.fitness(solution.representation)

    def calculate_path_distance(self, path):
        # Implementation for calculating path distance
        distance = 0
        # Convert path to numpy array if it isn't already
        path = np.array(path, dtype=int)
        # Close the loop by adding first city to end
        path_for_distance = np.append(path, path[0])
        weights = self.cities_graph.get_weights()
        for i in range(len(path_for_distance) - 1):
            # Convert 1-based indices to 0-based for array access
            city1 = int(path_for_distance[i]) - 1
            city2 = int(path_for_distance[i + 1]) - 1
            distance += weights[city1][city2]
        return distance

    def get_initial_population(self, population_size):
        """Generate initial population for GWO algorithm."""
        from Core.problem import Solution
        population = []

        for _ in range(population_size):
            # Create random permutation for TSP, ensuring city 1 is first
            remaining_cities = np.random.permutation(range(2, self.dim + 1))  # Cities 2 to dim
            route = np.concatenate([[1], remaining_cities])  # Start with city 1
            solution = Solution(route, self)
            solution.evaluate()
            population.append(solution)

        return population