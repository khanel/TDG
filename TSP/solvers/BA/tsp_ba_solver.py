"""
BA-based TSP Solver.

This module provides an implementation of a Bees Algorithm (BA)
solver for the Traveling Salesman Problem (TSP). It uses the base BA
framework with TSP-specific neighborhood operations.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from BA.BA import BeesAlgorithm
from TSP.TSP import TSPProblem, Graph
from Core.problem import Solution


class TSPBASolver:
    """
    Bees Algorithm solver for the Traveling Salesman Problem.

    This solver uses the Bees Algorithm with TSP-specific neighborhood
    operations including swap, insert, and reverse mutations.
    """

    def __init__(self, tsp_problem, population_size=50, m=10, e=3, nep=8, nsp=4,
                 ngh=3, max_iterations=200, stlim=None, verbosity=1,
                 exploration_boost=1.5, adaptive_exploration=True):
        """
        Initialize the TSP BA solver with enhanced exploration.

        Args:
            tsp_problem: TSPProblem instance
            population_size: Number of scout bees (n)
            m: Number of best sites selected for neighborhood search
            e: Elite sites among the selected sites
            nep: Recruited bees per elite site
            nsp: Recruited bees per non-elite selected site
            ngh: Neighborhood size (number of mutations per neighbor)
            max_iterations: Maximum number of iterations
            stlim: Stagnation limit for abandoning sites
            verbosity: Verbosity level
            exploration_boost: Multiplier for exploration enhancement
            adaptive_exploration: Whether to adapt exploration over time
        """
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbosity = verbosity

        # BA parameters
        self.m = m
        self.e = e
        self.nep = nep
        self.nsp = nsp
        self.ngh = ngh
        self.stlim = stlim

        # Enhanced exploration parameters
        self.exploration_boost = exploration_boost
        self.adaptive_exploration = adaptive_exploration

        # Visualization attributes
        self.live_fig = None
        self.live_ax = None
        self.best_fitness_history = []

    def setup_live_plot(self):
        """Initialize the live plotting window."""
        plt.ion()
        self.live_fig, self.live_ax = plt.subplots(figsize=(10, 10))
        self.live_ax.grid(True)
        self.live_ax.set_title('TSP Route (Live) - BA Solver')
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

        self.live_ax.set_title(f'Iteration {iteration}\nCurrent Distance: {best_fitness:.2f}')
        self.live_ax.axis('equal')
        self.live_ax.grid(True)
        self.live_ax.legend()

        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        plt.pause(0.01)

    def tsp_neighborhood_operator(self, solution, num_operations=1):
        """
        Generate a neighbor solution using TSP-specific operations.
        Uses only simple, guaranteed-valid operations.

        Args:
            solution: Current solution
            num_operations: Number of mutation operations to perform

        Returns:
            New solution representation (valid TSP tour)
        """
        # Start with a copy of the current solution
        new_repr = solution.representation.copy()
        n = len(new_repr)

        # Ensure city 1 is at the start
        if new_repr[0] != 1:
            city1_idx = new_repr.index(1)
            new_repr[0], new_repr[city1_idx] = new_repr[city1_idx], new_repr[0]

        # Perform simple swap operations only (most reliable for TSP)
        for _ in range(min(num_operations, n-2)):  # Limit operations to avoid over-modification
            if n > 2:
                # Swap two random cities (excluding city 1)
                i, j = np.random.choice(range(1, n), size=2, replace=False)
                new_repr[i], new_repr[j] = new_repr[j], new_repr[i]

        # Final validation
        if new_repr[0] != 1:
            city1_idx = new_repr.index(1)
            new_repr[0], new_repr[city1_idx] = new_repr[city1_idx], new_repr[0]

        return new_repr

    def solve(self):
        """
        Run the Bees Algorithm to solve the TSP problem.

        Returns:
            tuple: (best_solution, best_fitness)
        """
        print("Starting TSP solution with Bees Algorithm...")
        print(f"Number of cities: {len(self.tsp_problem.city_coords)}")
        print(f"Population size: {self.population_size}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Selected sites (m): {self.m}, Elite sites (e): {self.e}")
        print(f"Elite bees per site (nep): {self.nep}, Non-elite bees per site (nsp): {self.nsp}")

        # Initialize Bees Algorithm with enhanced exploration
        ba = BeesAlgorithm(
            problem=self.tsp_problem,
            population_size=self.population_size,
            m=self.m,
            e=self.e,
            nep=self.nep,
            nsp=self.nsp,
            ngh=self.ngh,
            max_iterations=self.max_iterations,
            stlim=self.stlim,
            verbosity=self.verbosity,
            exploration_boost=self.exploration_boost,
            adaptive_exploration=self.adaptive_exploration
        )

        # Setup visualization
        if self.verbosity >= 1:
            self.setup_live_plot()

        start_time = time.time()

        # Initialize population
        ba.initialize()
        print(f"Initial best fitness: {ba.best_solution.fitness:.2f}")

        # Override the neighborhood generation to use TSP-specific operations
        original_generate_neighbor = ba._generate_neighbor

        def tsp_generate_neighbor(site):
            """TSP-specific neighbor generation."""
            new_repr = self.tsp_neighborhood_operator(site, self.ngh)
            # Debug: check for invalid city indices
            max_city = len(self.tsp_problem.city_coords)
            if any(city > max_city or city < 1 for city in new_repr):
                print(f"Invalid cities in representation: {new_repr}")
                print(f"Max allowed city: {max_city}")
                # Return original if invalid
                return Solution(site.representation.copy(), self.tsp_problem)
            return Solution(new_repr, self.tsp_problem)

        ba._generate_neighbor = tsp_generate_neighbor

        # Run optimization
        for iteration in range(self.max_iterations):
            ba.step()

            # Update visualization
            if self.verbosity >= 1 and (iteration + 1) % 10 == 0:
                self.update_live_plot(ba.best_solution, iteration + 1, ba.best_solution.fitness)

            # Track convergence
            self.best_fitness_history.append(ba.best_solution.fitness)

            if self.verbosity >= 1 and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best fitness: {ba.best_solution.fitness:.2f}")

        end_time = time.time()

        # Final visualization
        if self.verbosity >= 1:
            self.update_live_plot(ba.best_solution, self.max_iterations, ba.best_solution.fitness)
            plt.pause(2)  # Show final result for 2 seconds

        # Print results
        print("\nOptimization complete!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Best fitness: {ba.best_solution.fitness:.2f}")
        print(f"Route: {' -> '.join(map(str, ba.best_solution.representation))} -> {ba.best_solution.representation[0]}")

        # Plot convergence history
        if self.verbosity >= 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.best_fitness_history)
            plt.title('BA TSP Solver - Convergence Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness (Tour Length)')
            plt.grid(True)
            plt.show()

        return ba.best_solution, ba.best_solution.fitness


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

    # Create and configure BA solver with enhanced exploration
    solver = TSPBASolver(
        tsp_problem=tsp_problem,
        population_size=100,
        m=12,      # Select 12 best sites (base value)
        e=10,      # 10 elite sites (base value)
        nep=10,    # 10 bees per elite site (base value)
        nsp=10,    # 10 bees per non-elite site (base value)
        ngh=20,    # 20 neighborhood operations per neighbor (base value)
        max_iterations=1500,
        stlim=10,  # Abandon sites after 10 iterations without improvement
        verbosity=1,
        exploration_boost=2.0,    # 2x exploration enhancement
        adaptive_exploration=True # Adaptive exploration over time
    )

    # Solve the problem
    best_solution, best_fitness = solver.solve()

    print("Final Results:")
    print(f"Best tour length: {best_fitness:.2f}")
    print(f"Best route: {' -> '.join(map(str, best_solution.representation))} -> {best_solution.representation[0]}")