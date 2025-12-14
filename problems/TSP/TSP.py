# Built-in Python packages
import sys
from pathlib import Path

# Third-party packages
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List

# Local project modules
# Ensure Core is in the Python path or adjust relative imports if needed
# Assuming Core is a top-level directory alongside TSP
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # Go up two levels from TSP/TSP.py to project root
from Core.problem import ProblemInterface, Solution

class TSPProblem(ProblemInterface):
    """
    Represents the Traveling Salesperson Problem (TSP).
    Conforms to the ProblemInterface for use with various search algorithms.
    """
    def __init__(self, graph: 'Graph', city_coords: List[tuple[float, float]]):
        """
        Initializes the TSP problem instance.

        Args:
            graph: A Graph object containing the distance matrix (weights).
            city_coords: A list of (x, y) coordinates for each city.
        """
        self.cities_graph = graph
        self.city_coords = city_coords

    def evaluate(self, solution: Solution) -> float:
        """
        Calculates the total distance of the tour represented by the solution.
        Lower distance is better fitness.

        Args:
            solution: The Solution object containing the tour (list of city indices).

        Returns:
            The total distance (fitness) of the tour.
        """
        return self.calculate_path_distance(solution.representation)

    def get_initial_solution(self) -> Solution:
        """
        Generates a single random initial solution (tour).
        The tour always starts at city 1 and visits other cities in a random order.

        Returns:
            A Solution object representing the initial tour.
        """
        vertices = self.cities_graph.get_vertices()
        # Always start with city 1, shuffle the rest
        remaining_cities = vertices[1:]  # All cities except city 1
        np.random.shuffle(remaining_cities)
        initial_tour = [1] + remaining_cities  # City 1 is always first
        solution = Solution(representation=initial_tour, problem=self)
        return solution

    def get_problem_info(self) -> Dict[str, Any]:
        """
        Provides essential information about the TSP instance.

        Returns:
            A dictionary containing the number of cities (dimension),
            problem type ('discrete'), and city coordinates.
        """
        return {
            'dimension': len(self.city_coords),
            'problem_type': 'discrete',
            'cities': self.city_coords # Provide coordinates for potential visualization
        }

    def calculate_path_distance(self, path_representation: List[int]) -> float:
        """
        Calculates the total distance of a given path (tour).
        Assumes the path is a list of city indices (1-based).
        The path implicitly returns to the start city to form a closed loop.

        Args:
            path_representation: A list of city indices representing the tour.

        Returns:
            The total distance of the tour.
        """
        distance = 0.0
        # Add the starting city to the end to close the loop for distance calculation
        path_for_distance = path_representation + [path_representation[0]]
        weights = self.cities_graph.get_weights()
        for i in range(len(path_representation)): # Iterate N times for N edges
            city1_idx = path_for_distance[i] - 1 # Convert 1-based city ID to 0-based index
            city2_idx = path_for_distance[i + 1] - 1
            distance += weights[city1_idx][city2_idx]
        return distance


class Graph:
    def __init__(self, weights):
        """Initialize a graph with a distance matrix.
        
        Args:
            weights (numpy.ndarray): Square matrix of distances between cities
        """
        self.weights = weights
        self.vertices = [i + 1 for i in range(len(weights))]
        self.edges = []
        self.calculate_edges(weights)

    def calculate_edges(self, weights, first_index=0):
        """Calculate edges and their weights from the distance matrix.
        
        Args:
            weights (numpy.ndarray): Distance matrix
            first_index (int): Starting index for edge calculation
        
        Returns:
            list: List of tuples (city1, city2, distance)
        """
        for i in range(first_index, len(self.vertices)):
            for j in range(len(self.vertices)):
                if i != j and weights[i][j] != 0:
                    self.edges.append((i + 1, j + 1, weights[i][j]))
        return self.edges

    def add_edge(self, weight):
        """Add a new city with its distances to existing cities.
        
        Args:
            weight (list): Distances from the new city to all existing cities
        
        Returns:
            list: Updated list of edges
        """
        self.vertices.append(len(self.vertices) + 1)
        for i, j in enumerate(weight):
            if weight[i] != 0:
                self.edges.append((i + 1, len(self.vertices), weight[i]))
                self.edges.append((len(self.vertices), i + 1, weight[i]))
        return self.edges

    def get_weights(self):
        """Get the distance matrix.
        
        Returns:
            numpy.ndarray: Matrix of distances between cities
        """
        return self.weights

    def get_vertices(self):
        """Get the list of cities.
        
        Returns:
            list: List of city indices (1-based)
        """
        return self.vertices
