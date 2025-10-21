"""
SA-based TSP Solver.

This module provides an implementation of a Simulated Annealing (SA)
solver for the Traveling Salesman Problem (TSP). It wraps the generic
SA engine with TSP-specific neighborhood operations that preserve a
valid permutation and keep city 1 as the fixed start for consistency
with other solvers.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from SA.SA import SimulatedAnnealing
from TSP.TSP import TSPProblem, Graph
from Core.problem import Solution


class TSPSASolver:
    """
    Simulated Annealing solver for the Traveling Salesman Problem.

    Uses a TSP-specific neighbor function (swap/2-opt variants) while
    keeping city 1 at the start of the tour for comparability.
    """

    def __init__(
        self,
        tsp_problem: TSPProblem,
        population_size: int = 10,
        *,
        initial_temp: float = 5.0,
        final_temp: float = 1e-3,
        alpha: float = 0.995,
        moves_per_temp: int = 5,
        max_iterations: int = 2000,
        verbosity: int = 1,
        two_opt_prob: float = 0.5,
    ):
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.moves_per_temp = moves_per_temp
        self.max_iterations = max_iterations
        self.verbosity = verbosity
        self.two_opt_prob = two_opt_prob

        # Visualization state
        self.live_fig = None
        self.live_ax = None
        self.best_fitness_history = []

    # --- Visualization helpers ---
    def setup_live_plot(self):
        plt.ion()
        self.live_fig, self.live_ax = plt.subplots(figsize=(10, 10))
        self.live_ax.grid(True)
        self.live_ax.set_title('TSP Route (Live) - SA Solver')
        plt.show()

    def update_live_plot(self, best_solution: Solution, iteration: int, best_fitness: float):
        if best_solution is None or self.live_ax is None:
            return

        self.live_ax.clear()
        problem_info = self.tsp_problem.get_problem_info()
        city_coords = problem_info.get('cities', [])

        # Plot cities
        for i, (x, y) in enumerate(city_coords, 1):
            self.live_ax.plot(x, y, 'ro', markersize=10)
            self.live_ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')

        # Plot route
        route = best_solution.representation
        n = len(route)
        for i in range(n):
            c1 = route[i] - 1
            c2 = route[(i + 1) % n] - 1
            if 0 <= c1 < len(city_coords) and 0 <= c2 < len(city_coords):
                x1, y1 = city_coords[c1]
                x2, y2 = city_coords[c2]
                self.live_ax.arrow(x1, y1, x2 - x1, y2 - y1,
                                   head_width=0.18, head_length=0.3,
                                   fc='blue', ec='blue', alpha=0.6)

        # Highlight start city
        if city_coords:
            sx, sy = city_coords[0]
            self.live_ax.plot(sx, sy, 'go', markersize=15, alpha=0.5, label='Start')

        self.live_ax.set_title(f'Iteration {iteration}\nCurrent Distance: {best_fitness:.2f}')
        self.live_ax.axis('equal')
        self.live_ax.grid(True)
        self.live_ax.legend()

        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        plt.pause(0.01)

    # --- TSP neighbor operators ---
    def _swap_neighbor(self, tour: list[int]) -> list[int]:
        n = len(tour)
        if n <= 3:
            return tour.copy()
        new = tour.copy()
        # swap excluding position 0 (keep city 1 fixed)
        i, j = np.random.choice(range(1, n), size=2, replace=False)
        new[i], new[j] = new[j], new[i]
        return new

    def _two_opt_neighbor(self, tour: list[int]) -> list[int]:
        n = len(tour)
        if n <= 4:
            return self._swap_neighbor(tour)
        i, j = sorted(np.random.choice(range(1, n), size=2, replace=False))
        new = tour[:i] + list(reversed(tour[i:j+1])) + tour[j+1:]
        # Ensure city 1 stays at front
        if new[0] != 1:
            k = new.index(1)
            new[0], new[k] = new[k], new[0]
        return new

    def tsp_neighbor(self, sol: Solution) -> Solution:
        tour = sol.representation
        if not isinstance(tour, list):
            tour = list(tour)
        # ensure city 1 at start
        if tour[0] != 1:
            idx1 = tour.index(1)
            tour[0], tour[idx1] = tour[idx1], tour[0]
        # choose operator
        if np.random.random() < self.two_opt_prob:
            new_tour = self._two_opt_neighbor(tour)
        else:
            new_tour = self._swap_neighbor(tour)
        return Solution(new_tour, self.tsp_problem)

    def solve(self):
        """Run Simulated Annealing to solve the TSP problem."""
        if self.verbosity >= 1:
            self.setup_live_plot()

        print("Starting TSP solution with Simulated Annealing...")
        print(f"Number of cities: {len(self.tsp_problem.city_coords)}")
        print(f"Population size: {self.population_size}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"T0: {self.initial_temp}, Tf: {self.final_temp}, alpha: {self.alpha}, moves/Temp: {self.moves_per_temp}")

        sa = SimulatedAnnealing(
            problem=self.tsp_problem,
            population_size=self.population_size,
            initial_temp=self.initial_temp,
            final_temp=self.final_temp,
            alpha=self.alpha,
            moves_per_temp=self.moves_per_temp,
            neighbor_fn=self.tsp_neighbor,
        )

        sa.initialize()
        print(f"Initial best fitness: {sa.best_solution.fitness:.2f}")

        start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            sa.step()

            # record and optionally show
            self.best_fitness_history.append(sa.best_solution.fitness)
            if self.verbosity >= 1 and iteration % 10 == 0:
                self.update_live_plot(sa.best_solution, iteration, sa.best_solution.fitness)
            if self.verbosity >= 1 and iteration % 50 == 0:
                print(f"Iteration {iteration}/{self.max_iterations}, Best fitness: {sa.best_solution.fitness:.2f}, T={sa.T:.4f}")

            # Optional early stop when cooled
            if sa.is_cooled():
                if self.verbosity >= 1:
                    print(f"Temperature cooled at iteration {iteration} (T={sa.T:.6f}).")
                break

        end_time = time.time()

        # Final visualization
        if self.verbosity >= 1:
            self.update_live_plot(sa.best_solution, iteration, sa.best_solution.fitness)
            plt.pause(2)

        print("\nOptimization complete!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Best fitness: {sa.best_solution.fitness:.2f}")
        route = sa.best_solution.representation
        print(f"Route: {' -> '.join(map(str, route))} -> {route[0]}")

        # Plot convergence history
        if self.verbosity >= 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.best_fitness_history)
            plt.title('SA TSP Solver - Convergence Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness (Tour Length)')
            plt.grid(True)
            plt.show()

        return sa.best_solution, sa.best_solution.fitness


if __name__ == "__main__":
    # Example usage with a random TSP instance
    num_cities = 120
    np.random.seed(42)

    # Create cities in a rough circle
    city_coords = []
    for i in range(num_cities):
        angle = 2 * np.pi * i / num_cities
        r = 5 + np.random.random() * 2
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        city_coords.append((x, y))

    # Distance matrix
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x1, y1 = city_coords[i]
                x2, y2 = city_coords[j]
                distances[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    graph = Graph(distances)
    tsp_problem = TSPProblem(graph, city_coords)

    solver = TSPSASolver(
        tsp_problem=tsp_problem,
        population_size=120,
        initial_temp=10.0,
        final_temp=1e-3,
        alpha=0.995,
        moves_per_temp=10,
        max_iterations=2000,
        verbosity=1,
        two_opt_prob=0.7,
    )

    best_solution, best_fitness = solver.solve()
    print("\nFinal Results:")
    print(f"Best tour length: {best_fitness:.2f}")
    print(f"Best route: {' -> '.join(map(str, best_solution.representation))} -> {best_solution.representation[0]}")

