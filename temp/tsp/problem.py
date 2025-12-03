
"""
This file contains the self-contained implementation of the TSP problem,
including the core problem logic and the adapter required to interface
with the reinforcement learning environment.
"""

from typing import Dict, Any, List, Optional, Iterable
import numpy as np

# We need the base classes from core.py
from temp.core.base import ProblemInterface, Solution


class Graph:
    """A simple graph representation for the TSP."""
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)
        self.num_vertices = len(self.weights)
        self.vertices = list(range(self.num_vertices))

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_vertices(self) -> List[int]:
        return self.vertices


class TSPProblem(ProblemInterface):
    """Represents the Traveling Salesperson Problem."""
    def __init__(self, graph: Graph, city_coords: List[tuple[float, float]]):
        self.cities_graph = graph
        self.city_coords = city_coords
        self.num_cities = len(city_coords)

    def evaluate(self, solution: Solution) -> float:
        """Calculates the total distance of the tour with NumPy optimization."""
        path = np.asarray(solution.representation, dtype=int)
        weights = self.cities_graph.get_weights()
        
        # Vectorized distance calculation using advanced indexing
        u = path
        v = np.roll(path, -1)  # Shift to get next city for each edge
        distance = np.sum(weights[u, v])
        
        return distance

    def get_initial_solution(self) -> Solution:
        """Generates a random tour."""
        tour = np.random.permutation(self.num_cities).tolist()
        return Solution(representation=tour, problem=self)

    def get_initial_population(self, population_size: int) -> List['Solution']:
        """
        Generates an initial population of tours using vectorized operations.
        """
        # Vectorized population generation
        tours = np.array([np.random.permutation(self.num_cities) for _ in range(population_size)])
        population = [Solution(tour.tolist(), self) for tour in tours]
        
        # Evaluate all solutions
        for sol in population:
            sol.evaluate()
        
        return population

    def get_problem_info(self) -> Dict[str, Any]:
        """Provides essential information about the TSP instance."""
        return {
            'dimension': self.num_cities,
            'problem_type': 'permutation',
            'cities': self.city_coords,
        }


class TSPAdapter(ProblemInterface):
    """Adapter for TSP problems with optional randomization."""
    def __init__(self, num_cities: int = 50, seed: Optional[int] = 42, grid_size: float = 100.0):
        self._num_cities = num_cities
        self._grid_size = grid_size
        self._rng = np.random.default_rng(seed)
        self._rebuild_problem()

    def evaluate(self, solution: Solution) -> float:
        return self.tsp_problem.evaluate(solution)

    def get_initial_solution(self) -> Solution:
        return self.tsp_problem.get_initial_solution()

    def get_problem_info(self) -> Dict[str, Any]:
        info = self.tsp_problem.get_problem_info()
        # Estimate bounds for normalization in the observation space
        info["lower_bound"] = self._grid_size * np.sqrt(self._num_cities)
        info["upper_bound"] = self._grid_size * self._num_cities * 1.5
        return info

    def get_initial_population(self, population_size: int) -> List['Solution']:
        """Generate initial population of random tours."""
        return self.tsp_problem.get_initial_population(population_size)

    def get_city_coords(self) -> np.ndarray:
        """Return city coordinates for visualization."""
        return np.array(self.tsp_problem.city_coords)

    def _rebuild_problem(self):
        """Creates a new TSP instance."""
        coords = self._rng.random((self._num_cities, 2)) * self._grid_size
        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        
        graph = Graph(dist_matrix)
        self.tsp_problem = TSPProblem(graph, coords.tolist())

