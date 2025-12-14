"""
PSO-based TSP Solver.

This module provides an implementation of a Particle Swarm Optimization (PSO)
solver for the Traveling Salesman Problem (TSP). It adapts PSO for discrete
permutation-based optimization with TSP-specific operations.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.PSO.PSO import ParticleSwarmOptimization
from problems.TSP.TSP import TSPProblem, Graph
from Core.problem import Solution


class TSPPSOSolver:
    """
    Particle Swarm Optimization solver for the Traveling Salesman Problem.

    This solver adapts PSO for discrete permutation optimization using
    TSP-specific position updates and neighborhood operations.
    """

    def __init__(self, tsp_problem, population_size=30, omega=0.7, c1=1.5, c2=1.5,
                 max_iterations=200, use_constriction=False, verbosity=1,
                 exploration_boost=1.5, adaptive_exploration=True):
        """
        Initialize the TSP PSO solver.

        Args:
            tsp_problem: TSPProblem instance
            population_size: Number of particles in swarm
            omega: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            max_iterations: Maximum number of iterations
            use_constriction: Whether to use constriction factor
            verbosity: Verbosity level
            exploration_boost: Multiplier for exploration enhancement
            adaptive_exploration: Whether to adapt exploration over time
        """
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbosity = verbosity

        # PSO parameters
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.use_constriction = use_constriction

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
        self.live_ax.set_title('TSP Route (Live) - PSO Solver')
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

    def discrete_pso_update(self, position, velocity, personal_best, global_best):
        """
        Perform discrete PSO update for TSP using permutation-based operations.

        Args:
            position: Current particle position (TSP tour)
            velocity: Current particle velocity
            personal_best: Particle's personal best position
            global_best: Swarm's global best position

        Returns:
            Updated position and velocity
        """
        n = len(position)

        # Convert positions to numpy arrays for easier manipulation
        pos = np.array(position)
        pbest = np.array(personal_best)
        gbest = np.array(global_best)

        # Calculate "velocity" as probability of following different influences
        r1 = np.random.random()
        r2 = np.random.random()

        # Create new position by combining influences
        new_pos = pos.copy()

        # Apply cognitive influence (personal best)
        if r1 < self.c1 / (self.c1 + self.c2):
            # Swap elements to move toward personal best
            diff_positions = []
            for i in range(n):
                if pos[i] != pbest[i]:
                    diff_positions.append(i)

            if len(diff_positions) >= 2:
                # Swap a random pair of different positions
                i, j = np.random.choice(diff_positions, size=2, replace=False)
                new_pos[i], new_pos[j] = new_pos[j], new_pos[i]

        # Apply social influence (global best)
        if r2 < self.c2 / (self.c1 + self.c2):
            # Swap elements to move toward global best
            diff_positions = []
            for i in range(n):
                if pos[i] != gbest[i]:
                    diff_positions.append(i)

            if len(diff_positions) >= 2:
                # Swap a random pair of different positions
                i, j = np.random.choice(diff_positions, size=2, replace=False)
                new_pos[i], new_pos[j] = new_pos[j], new_pos[i]

        # Apply inertia (tendency to stay in current position)
        if np.random.random() > self.omega:
            # Apply a small random change
            if n > 2:
                i, j = np.random.choice(range(1, n), size=2, replace=False)
                new_pos[i], new_pos[j] = new_pos[j], new_pos[i]

        # Ensure city 1 stays at the start
        if new_pos[0] != 1:
            city1_idx = np.where(new_pos == 1)[0][0]
            new_pos[0], new_pos[city1_idx] = new_pos[city1_idx], new_pos[0]

        return new_pos.tolist(), velocity  # Return velocity unchanged for discrete PSO

    def solve(self):
        """
        Run the PSO to solve the TSP problem.

        Returns:
            tuple: (best_solution, best_fitness)
        """
        print("Starting TSP solution with Particle Swarm Optimization...")
        print(f"Number of cities: {len(self.tsp_problem.city_coords)}")
        print(f"Population size: {self.population_size}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Omega: {self.omega}, C1: {self.c1}, C2: {self.c2}")
        print(f"Constriction: {self.use_constriction}")

        # Setup visualization
        if self.verbosity >= 1:
            self.setup_live_plot()

        start_time = time.time()

        # Initialize swarm manually for TSP
        swarm_positions = []
        swarm_velocities = []
        personal_bests = []
        personal_best_fitness = []

        for _ in range(self.population_size):
            # Generate random TSP tour
            initial_solution = self.tsp_problem.get_initial_solution()
            initial_solution.evaluate()  # Ensure fitness is calculated
            swarm_positions.append(initial_solution.representation)

            # Initialize velocity (for discrete PSO, this can be minimal)
            velocity = [0.0] * len(initial_solution.representation)
            swarm_velocities.append(velocity)

            # Initialize personal best
            personal_bests.append(initial_solution.representation.copy())
            personal_best_fitness.append(initial_solution.fitness)

        # Find initial global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_bests[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]

        print(f"Initial best fitness: {global_best_fitness:.2f}")

        # Main PSO loop
        for iteration in range(self.max_iterations):
            # Update each particle
            for i in range(self.population_size):
                # Perform discrete PSO update
                new_position, new_velocity = self.discrete_pso_update(
                    swarm_positions[i],
                    swarm_velocities[i],
                    personal_bests[i],
                    global_best
                )

                # Create solution and evaluate
                solution = Solution(new_position, self.tsp_problem)
                solution.evaluate()

                # Update personal best
                if solution.fitness < personal_best_fitness[i]:
                    personal_bests[i] = new_position.copy()
                    personal_best_fitness[i] = solution.fitness

                # Update position and velocity
                swarm_positions[i] = new_position
                swarm_velocities[i] = new_velocity

            # Update global best
            current_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_best_idx] < global_best_fitness:
                global_best = personal_bests[current_best_idx].copy()
                global_best_fitness = personal_best_fitness[current_best_idx]

            # Update visualization
            if self.verbosity >= 1 and (iteration + 1) % 10 == 0:
                best_solution = Solution(global_best, self.tsp_problem)
                best_solution.fitness = global_best_fitness
                self.update_live_plot(best_solution, iteration + 1, global_best_fitness)

            # Track convergence
            self.best_fitness_history.append(global_best_fitness)

            if self.verbosity >= 1 and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best fitness: {global_best_fitness:.2f}")

        end_time = time.time()

        # Final visualization
        if self.verbosity >= 1:
            best_solution = Solution(global_best, self.tsp_problem)
            best_solution.fitness = global_best_fitness
            self.update_live_plot(best_solution, self.max_iterations, global_best_fitness)
            plt.pause(2)  # Show final result for 2 seconds

        # Print results
        print("\nOptimization complete!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Best fitness: {global_best_fitness:.2f}")
        print(f"Route: {' -> '.join(map(str, global_best))} -> {global_best[0]}")

        # Plot convergence history
        if self.verbosity >= 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.best_fitness_history)
            plt.title('PSO TSP Solver - Convergence Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness (Tour Length)')
            plt.grid(True)
            plt.show()

        best_solution = Solution(global_best, self.tsp_problem)
        best_solution.fitness = global_best_fitness
        return best_solution, global_best_fitness


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

    # Create and configure PSO solver
    solver = TSPPSOSolver(
        tsp_problem=tsp_problem,
        population_size=30,
        omega=0.7,
        c1=1.5,
        c2=1.5,
        max_iterations=2000,
        use_constriction=False,
        verbosity=1,
        exploration_boost=1.5,
        adaptive_exploration=True
    )

    # Solve the problem
    best_solution, best_fitness = solver.solve()

    print("\nFinal Results:")
    print(f"Best tour length: {best_fitness:.2f}")
    print(f"Best route: {' -> '.join(map(str, best_solution.representation))} -> {best_solution.representation[0]}")