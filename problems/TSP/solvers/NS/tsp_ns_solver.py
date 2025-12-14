"""
NS-based TSP Solver.

This module provides an implementation of a Novelty Search (NS)
solver for the Traveling Salesman Problem (TSP). It uses behavioral
diversity rather than objective optimization to find novel tour patterns.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from NS.NS import NoveltySearch
from problems.TSP.TSP import TSPProblem, Graph
from Core.problem import Solution
from NS.Problem import tsp_behavior_characterization, tsp_reproduction_operator


class TSPNSSolver:
    """
    Novelty Search solver for the Traveling Salesman Problem.

    This solver uses Novelty Search to discover diverse and novel TSP tour patterns.
    Instead of optimizing for tour length, it optimizes for behavioral novelty,
    leading to exploration of different solution spaces.
    """

    def __init__(self, tsp_problem, population_size=50, k_neighbors=15,
                 archive_threshold=0.0, max_archive_size=None, max_iterations=200,
                 verbosity=1):
        """
        Initialize the TSP NS solver.

        Args:
            tsp_problem: TSPProblem instance
            population_size: Number of individuals in population
            k_neighbors: Number of nearest neighbors for novelty computation
            archive_threshold: Threshold for adding behaviors to archive
            max_archive_size: Maximum archive size (None = unlimited)
            max_iterations: Maximum number of iterations
            verbosity: Verbosity level
        """
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbosity = verbosity

        # NS parameters
        self.k_neighbors = k_neighbors
        self.archive_threshold = archive_threshold
        self.max_archive_size = max_archive_size

        # Visualization attributes
        self.live_fig = None
        self.live_ax = None
        self.best_fitness_history = []
        self.novelty_history = []

    def setup_live_plot(self):
        """Initialize the live plotting window."""
        plt.ion()
        self.live_fig, self.live_ax = plt.subplots(figsize=(10, 10))
        self.live_ax.grid(True)
        self.live_ax.set_title('TSP Route (Live) - NS Solver')
        plt.show()

    def update_live_plot(self, best_solution, iteration, best_fitness):
        """Update the live plot with current best solution."""
        if best_solution is None or self.live_ax is None:
            return

        self.live_ax.clear()

        # Plot cities
        problem_info = self.tsp_problem.get_problem_info()
        city_coords = problem_info.get('cities', [])
        for i, (x, y) in enumerate(city_coords, 1):
            self.live_ax.plot(x, y, 'ro', markersize=10)
            self.live_ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')

        # Plot route
        route = best_solution.representation
        n = len(route)
        for i in range(n):
            city1 = route[i] - 1  # Convert to 0-based indexing
            city2 = route[(i + 1) % n] - 1
            if city1 < len(city_coords) and city2 < len(city_coords):
                x1, y1 = city_coords[city1]
                x2, y2 = city_coords[city2]
                self.live_ax.arrow(x1, y1, x2 - x1, y2 - y1,
                                 head_width=0.18, head_length=0.3,
                                 fc='blue', ec='blue', alpha=0.6)

        # Highlight start city
        if city_coords:
            start_x, start_y = city_coords[0]
            self.live_ax.plot(start_x, start_y, 'go', markersize=15, alpha=0.5, label='Start')

        self.live_ax.set_title(f'Iteration {iteration}\nFitness: {best_fitness:.2f}')
        self.live_ax.axis('equal')
        self.live_ax.grid(True)
        self.live_ax.legend()

        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        plt.pause(0.01)

    def solve(self):
        """
        Run the Novelty Search to solve the TSP problem.

        Returns:
            tuple: (best_solution, best_fitness)
        """
        print("Starting TSP solution with Novelty Search...")
        print(f"Number of cities: {len(self.tsp_problem.city_coords)}")
        print(f"Population size: {self.population_size}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"k-neighbors: {self.k_neighbors}, Archive threshold: {self.archive_threshold}")

        # Setup visualization
        if self.verbosity >= 1:
            self.setup_live_plot()

        start_time = time.time()

        # Initialize Novelty Search with TSP-specific functions
        ns = NoveltySearch(
            problem=self.tsp_problem,
            population_size=self.population_size,
            k_neighbors=self.k_neighbors,
            archive_threshold=self.archive_threshold,
            max_archive_size=self.max_archive_size,
            behavior_characterization=tsp_behavior_characterization,
            reproduction_operator=tsp_reproduction_operator,
            verbosity=self.verbosity
        )

        # Initialize population
        ns.initialize()

        print(f"Initial archive size: {len(ns.archive)}")

        # Main NS loop
        for iteration in range(self.max_iterations):
            # Perform one generation
            ns.step()

            # Track metrics
            archive_info = ns.get_archive_info()
            diversity_info = ns.get_diversity_metrics()

            self.best_fitness_history.append(ns.best_solution.fitness)
            self.novelty_history.append(archive_info['avg_novelty'])

            # Update visualization
            if self.verbosity >= 1 and (iteration + 1) % 10 == 0:
                self.update_live_plot(ns.best_solution, iteration + 1, ns.best_solution.fitness)

            if self.verbosity >= 1 and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"  Archive size: {archive_info['archive_size']}")
                print(f"  Avg novelty: {archive_info['avg_novelty']:.4f}")
                print(f"  Best fitness: {ns.best_solution.fitness:.2f}")
                print(f"  Behavioral diversity: {diversity_info['behavioral_diversity']:.4f}")

        end_time = time.time()

        # Final visualization
        if self.verbosity >= 1:
            self.update_live_plot(ns.best_solution, self.max_iterations, ns.best_solution.fitness)
            plt.pause(2)  # Show final result for 2 seconds

        # Print results
        print("\nOptimization complete!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Final archive size: {len(ns.archive)}")
        print(f"Best fitness: {ns.best_solution.fitness:.2f}")
        print(f"Route: {' -> '.join(map(str, ns.best_solution.representation))} -> {ns.best_solution.representation[0]}")

        # Final diversity analysis
        final_diversity = ns.get_diversity_metrics()
        print("\nBehavioral Diversity Metrics:")
        print(f"  Behavioral diversity: {final_diversity['behavioral_diversity']:.4f}")
        print(f"  Unique behaviors: {final_diversity['unique_behaviors']}")

        # Plot convergence history
        if self.verbosity >= 1:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(self.best_fitness_history)
            plt.title('NS TSP Solver - Fitness Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness (Tour Length)')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(self.novelty_history)
            plt.title('NS TSP Solver - Novelty Evolution')
            plt.xlabel('Iteration')
            plt.ylabel('Average Novelty Score')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        return ns.best_solution, ns.best_solution.fitness

    def analyze_archive_diversity(self, ns_instance):
        """
        Analyze the diversity of behaviors in the archive.

        Args:
            ns_instance: Trained NoveltySearch instance

        Returns:
            Dictionary with diversity analysis
        """
        if not ns_instance.archive:
            return {}

        archive_behaviors = np.array(ns_instance.archive)

        # Compute pairwise distances
        distances = []
        for i in range(len(archive_behaviors)):
            for j in range(i + 1, len(archive_behaviors)):
                dist = np.linalg.norm(archive_behaviors[i] - archive_behaviors[j])
                distances.append(dist)

        return {
            'archive_size': len(ns_instance.archive),
            'avg_behavior_distance': np.mean(distances) if distances else 0.0,
            'max_behavior_distance': np.max(distances) if distances else 0.0,
            'behavior_spread': np.std(distances) if distances else 0.0
        }


if __name__ == "__main__":
    # Create a sample TSP problem
    num_cities = 20
    np.random.seed(42)  # For reproducibility

    # Create cities in a circle with some random displacement
    city_coords = []
    for i in range(num_cities):
        angle = 2 * np.pi * i / num_cities
        r = 5 + np.random.random() * 2  # Random radius between 5 and 7
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        city_coords.append((x, y))

    # Calculate distance matrix
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x1, y1 = city_coords[i]
                x2, y2 = city_coords[j]
                distances[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Create graph and TSP problem
    graph = Graph(distances)
    tsp_problem = TSPProblem(graph, city_coords)

    # Create and configure NS solver
    solver = TSPNSSolver(
        tsp_problem=tsp_problem,
        population_size=50,
        k_neighbors=15,
        archive_threshold=0.0,  # Archive all novel behaviors
        max_archive_size=500,   # Limit archive size
        max_iterations=200,
        verbosity=1
    )

    # Solve the problem
    best_solution, best_fitness = solver.solve()

    print("\nFinal Results:")
    print(f"Best tour length: {best_fitness:.2f}")
    print(f"Best route: {' -> '.join(map(str, best_solution.representation))} -> {best_solution.representation[0]}")