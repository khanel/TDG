"""
GBO-based TSP Solver.

This module provides an implementation of a Gradient-Based Optimizer (GBO)
solver for the Traveling Salesman Problem (TSP). It adapts GBO for discrete
permutation-based optimization with TSP-specific operations.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.GBO.GBO import GradientBasedOptimizer
from problems.TSP.TSP import TSPProblem, Graph
from Core.problem import Solution


class TSPGBOSolver:
    """
    Gradient-Based Optimizer solver for the Traveling Salesman Problem.

    This solver adapts GBO for discrete permutation optimization using
    TSP-specific GSR and LEO operations.
    """

    def __init__(self, tsp_problem, population_size=30, alpha=1.0, beta=1.0,
                 leo_prob=0.1, max_iterations=200, step_size=1.0, verbosity=1,
                 exploration_boost=1.5, adaptive_exploration=True):
        """
        Initialize the TSP GBO solver.

        Args:
            tsp_problem: TSPProblem instance
            population_size: Number of agents in population
            alpha: GSR coefficient for global best influence
            beta: GSR coefficient for population influence
            leo_prob: Probability of applying Local Escaping Operator
            max_iterations: Maximum number of iterations
            step_size: Step size for GSR movements
            verbosity: Verbosity level
            exploration_boost: Multiplier for exploration enhancement
            adaptive_exploration: Whether to adapt exploration over time
        """
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbosity = verbosity

        # GBO parameters
        self.alpha = alpha
        self.beta = beta
        self.leo_prob = leo_prob
        self.step_size = step_size

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
        self.live_ax.set_title('TSP Route (Live) - GBO Solver')
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

    def discrete_gsr_update(self, position, global_best, reference_agents):
        """
        Apply discrete GSR update for TSP.

        Args:
            position: Current agent position (TSP tour)
            global_best: Global best position
            reference_agents: List of reference agent positions

        Returns:
            Updated position
        """
        n = len(position)

        # Convert to numpy arrays for easier manipulation
        pos = np.array(position)
        gbest = np.array(global_best)

        # Select reference agents
        if len(reference_agents) >= 2:
            r1_pos = np.array(reference_agents[0])
            r2_pos = np.array(reference_agents[1])
        else:
            # Use global best as fallback
            r1_pos = gbest
            r2_pos = gbest

        # Build direction vector using discrete operations
        new_pos = pos.copy()

        # Find positions that differ from global best
        diff_positions = []
        for i in range(n):
            if pos[i] != gbest[i]:
                diff_positions.append(i)

        # Apply GSR-inspired movements
        if len(diff_positions) > 0:
            # Select a subset of positions to modify
            num_changes = max(1, int(len(diff_positions) * self.step_size))
            positions_to_change = np.random.choice(diff_positions,
                                                 size=min(num_changes, len(diff_positions)),
                                                 replace=False)

            for pos_idx in positions_to_change:
                # Find where the correct city is in current position
                correct_city = gbest[pos_idx]
                locations = np.where(pos == correct_city)[0]

                if len(locations) > 0:
                    current_location = locations[0]

                    if current_location != pos_idx:
                        # Swap to move correct city to right position
                        new_pos[pos_idx], new_pos[current_location] = new_pos[current_location], new_pos[pos_idx]
                # If city is not found in current position, skip this change
                # This can happen due to previous modifications

        return new_pos.tolist()

    def tsp_leo_operator(self, position, global_best):
        """
        Apply TSP-specific Local Escaping Operator.

        Args:
            position: Current position (TSP tour)
            global_best: Global best position

        Returns:
            Escaped position
        """
        n = len(position)
        new_pos = position.copy()

        # Multiple escape strategies for TSP
        escape_type = np.random.choice(['segment_reverse', 'city_relocate', 'segment_swap'], p=[0.4, 0.3, 0.3])

        if escape_type == 'segment_reverse' and n > 3:
            # Reverse a random segment (excluding city 1)
            i, j = sorted(np.random.choice(range(1, n), size=2, replace=False))
            if j - i > 1:
                new_pos[i:j+1] = new_pos[i:j+1][::-1]

        elif escape_type == 'city_relocate' and n > 2:
            # Move a city to a different position
            i, j = np.random.choice(range(1, n), size=2, replace=False)
            city = new_pos[i]
            # Remove from current position
            del new_pos[i]
            # Insert at new position
            new_pos.insert(j, city)

        elif escape_type == 'segment_swap' and n > 4:
            # Swap two segments
            # Choose two non-overlapping segments
            start1 = np.random.randint(1, n-2)
            length1 = np.random.randint(1, min(3, n-start1-1))
            end1 = start1 + length1

            # Find valid start for second segment
            min_start2 = end1 + 1
            if min_start2 < n - 1:
                start2 = np.random.randint(min_start2, n-1)
                length2 = np.random.randint(1, min(3, n-start2))
                end2 = start2 + length2

                if end2 < n:
                    # Swap the segments
                    segment1 = new_pos[start1:end1+1]
                    segment2 = new_pos[start2:end2+1]
                    new_pos[start1:end1+1] = segment2
                    new_pos[start2:end2+1] = segment1

        # Ensure city 1 is still at the start
        if new_pos[0] != 1:
            city1_idx = new_pos.index(1)
            new_pos[0], new_pos[city1_idx] = new_pos[city1_idx], new_pos[0]

        return new_pos

    def solve(self):
        """
        Run the GBO to solve the TSP problem.

        Returns:
            tuple: (best_solution, best_fitness)
        """
        print("Starting TSP solution with Gradient-Based Optimizer...")
        print(f"Number of cities: {len(self.tsp_problem.city_coords)}")
        print(f"Population size: {self.population_size}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Alpha: {self.alpha}, Beta: {self.beta}, LEO Prob: {self.leo_prob}")

        # Setup visualization
        if self.verbosity >= 1:
            self.setup_live_plot()

        start_time = time.time()

        # Initialize population manually for TSP
        swarm_positions = []
        global_best = None
        global_best_fitness = float('inf')

        for _ in range(self.population_size):
            # Generate random TSP tour
            initial_solution = self.tsp_problem.get_initial_solution()
            initial_solution.evaluate()
            swarm_positions.append(initial_solution.representation)

            # Update global best
            if initial_solution.fitness < global_best_fitness:
                global_best = initial_solution.representation.copy()
                global_best_fitness = initial_solution.fitness

        print(f"Initial best fitness: {global_best_fitness:.2f}")

        # Main GBO loop
        for iteration in range(self.max_iterations):
            new_positions = []

            for i in range(self.population_size):
                current_pos = swarm_positions[i]

                # Select reference agents (different from current)
                available_indices = [j for j in range(self.population_size) if j != i]
                if len(available_indices) >= 2:
                    ref_indices = np.random.choice(available_indices, size=2, replace=False)
                    reference_agents = [swarm_positions[ref_indices[0]], swarm_positions[ref_indices[1]]]
                else:
                    reference_agents = [global_best, global_best]

                # Apply GSR (Gradient Search Rule)
                new_pos = self.discrete_gsr_update(current_pos, global_best, reference_agents)

                # Apply LEO (Local Escaping Operator)
                if np.random.random() < self.leo_prob:
                    new_pos = self.tsp_leo_operator(new_pos, global_best)

                # Create solution and evaluate
                solution = Solution(new_pos, self.tsp_problem)
                solution.evaluate()

                # Selection: keep the better solution
                current_solution = Solution(current_pos, self.tsp_problem)
                current_solution.evaluate()

                if solution.fitness < current_solution.fitness:
                    new_positions.append(new_pos)
                else:
                    new_positions.append(current_pos)

                # Update global best
                if solution.fitness < global_best_fitness:
                    global_best = new_pos.copy()
                    global_best_fitness = solution.fitness

            swarm_positions = new_positions

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
            plt.title('GBO TSP Solver - Convergence Curve')
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

    # Create and configure GBO solver
    solver = TSPGBOSolver(
        tsp_problem=tsp_problem,
        population_size=30,
        alpha=1.0,
        beta=1.0,
        leo_prob=0.1,
        max_iterations=2000,
        step_size=1.0,
        verbosity=1,
        exploration_boost=1.5,
        adaptive_exploration=True
    )

    # Solve the problem
    best_solution, best_fitness = solver.solve()

    print("\nFinal Results:")
    print(f"Best tour length: {best_fitness:.2f}")
    print(f"Best route: {' -> '.join(map(str, best_solution.representation))} -> {best_solution.representation[0]}")