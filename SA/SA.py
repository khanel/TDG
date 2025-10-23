import math
import random
import numpy as np
from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm

class SimulatedAnnealing(SearchAlgorithm):
    """
    Simulated Annealing (SA) search algorithm.

    This algorithm is a metaheuristic inspired by the annealing process in metallurgy.
    It explores the search space by iteratively moving from a current solution to a
    neighboring one. Moves to better solutions are always accepted, while moves to
    worse solutions are accepted with a probability that decreases over time,
    allowing the algorithm to escape local optima.

    Args:
        problem (ProblemInterface): The optimization problem to solve.
        initial_temperature (float): The starting temperature.
        cooling_rate (float): The rate at which the temperature decreases (e.g., 0.95).
        max_iterations (int): The total number of iterations to run.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, problem: ProblemInterface, initial_temperature: float = 1000.0,
                 final_temperature: float = 1e-3, cooling_rate: float = 0.99,
                 moves_per_temp: int = 1, max_iterations: int = 10000,
                 neighbor_fn=None, population_size: int = 1, **kwargs):
        # SA is a single-state search, so population_size defaults to 1 but can be overridden
        super().__init__(problem, population_size=population_size, **kwargs)

        if not (0 < cooling_rate < 1):
            raise ValueError("Cooling rate must be between 0 and 1.")
        if initial_temperature <= final_temperature:
            raise ValueError("Initial temperature must be greater than final temperature.")
        if moves_per_temp < 1:
            raise ValueError("Moves per temperature must be at least 1.")

        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.moves_per_temp = moves_per_temp
        self.max_iterations = max_iterations
        self.neighbor_fn = neighbor_fn  # Custom neighbor function, if provided

        # The 'population' will hold our single current state
        self.current_solution: Solution = None

    def initialize(self):
        """
        Initializes the algorithm by creating a single random solution.
        """
        super().initialize()
        # The base class creates a population list; we work with a single solution
        self.current_solution = self.population[0]
        if self.best_solution is None or self.current_solution.fitness < self.best_solution.fitness:
            self.best_solution = self.current_solution.copy()

    def step(self):
        """
        Performs a single iteration of the Simulated Annealing algorithm.

        In each temperature level, performs moves_per_temp moves before cooling.
        """
        if self.iteration >= self.max_iterations:
            return

        # Perform moves_per_temp moves at current temperature
        for _ in range(self.moves_per_temp):
            if self.iteration >= self.max_iterations:
                break

            # 1. Generate a neighbor
            if self.neighbor_fn is not None:
                neighbor = self.neighbor_fn(self.current_solution)
            else:
                neighbor = self._create_neighbor(self.current_solution)
            neighbor.evaluate()

            # 2. Decide whether to accept the neighbor
            cost_diff = neighbor.fitness - self.current_solution.fitness

            if cost_diff < 0 or random.uniform(0, 1) < self._acceptance_probability(cost_diff):
                self.current_solution = neighbor
                # Update the population list for compatibility with the base class
                self.population[0] = self.current_solution

            # Update the best solution found so far
            if self.current_solution.fitness < self.best_solution.fitness:
                self.best_solution = self.current_solution.copy()

            self.iteration += 1

        # 3. Cool down the temperature after moves_per_temp moves
        self._cool_down()

    def _create_neighbor(self, solution: Solution) -> Solution:
        """
        Creates a neighboring solution by making a small, random modification.
        The modification strategy depends on the problem's representation type.
        """
        problem_info = self.problem.get_problem_info()
        representation_type = problem_info.get("problem_type", "continuous")

        neighbor_representation = solution.representation.copy()

        if representation_type == "continuous":
            # For continuous problems, iterate through dimensions and perturb with some probability
            dim = problem_info.get("dimension", len(neighbor_representation))
            lower_bounds = problem_info.get("lower_bounds")
            upper_bounds = problem_info.get("upper_bounds")
            
            # Probability of modifying a single dimension
            modification_prob = 1.0 / dim 

            for i in range(dim):
                if random.uniform(0, 1) < modification_prob:
                    # Scale the.perturbation by the current temperature
                    perturbation_scale = self.temperature / self.initial_temperature
                    change = np.random.normal(0, perturbation_scale) # Smaller changes as temp decreases
                    
                    neighbor_representation[i] += change
                    
                    # Clamp the value to the defined bounds
                    if lower_bounds is not None and upper_bounds is not None:
                        neighbor_representation[i] = np.clip(
                            neighbor_representation[i],
                            lower_bounds[i],
                            upper_bounds[i]
                        )

        elif representation_type == "discrete" or isinstance(neighbor_representation, list):
            # For discrete problems (e.g., permutations for TSP), swap two elements
            if len(neighbor_representation) > 1:
                i, j = random.sample(range(len(neighbor_representation)), 2)
                neighbor_representation[i], neighbor_representation[j] = neighbor_representation[j], neighbor_representation[i]
        
        elif isinstance(neighbor_representation, np.ndarray) and neighbor_representation.dtype == bool:
            # For binary string problems (e.g., Knapsack), flip a random bit
            idx_to_flip = random.randint(0, len(neighbor_representation) - 1)
            neighbor_representation[idx_to_flip] = not neighbor_representation[idx_to_flip]
            
        else:
            # Fallback for unsupported types - could be extended
            raise TypeError(f"Unsupported representation type for neighbor generation: {type(neighbor_representation)}")

        return Solution(neighbor_representation, self.problem)

    def _acceptance_probability(self, cost_difference: float) -> float:
        """
        Calculates the probability of accepting a worse solution.
        The probability is e^(-delta_cost / temperature).
        """
        if self.temperature > 0:
            try:
                return math.exp(-cost_difference / self.temperature)
            except OverflowError:
                return float('inf') # Should be handled by the random.uniform check
        return 0.0

    def _cool_down(self):
        """
        Reduces the temperature based on the cooling schedule.
        """
        self.temperature *= self.cooling_rate
        if self.temperature < self.final_temperature:
            self.temperature = self.final_temperature

    def is_cooled(self) -> bool:
        """
        Checks if the temperature has reached the final temperature.
        """
        return self.temperature <= self.final_temperature

    def get_state(self) -> dict:
        """Returns the current state of the algorithm for logging or monitoring."""
        state = super().get_state() if hasattr(super(), 'get_state') else {}
        state.update({
            "temperature": self.temperature,
            "current_fitness": self.current_solution.fitness if self.current_solution else None,
            "is_cooled": self.is_cooled(),
        })
        return state