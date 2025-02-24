"""
GWO-based TSP Solver.

This module provides an implementation of a Gray Wolf Optimization (GWO)
solver for the Traveling Salesman Problem (TSP). It extends the base GWO
framework with TSP-specific operations and visualizations.
"""
import sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from GWO.GWO import GrayWolfOptimization
from GWO.Problem import GWO_TSPProblem
from TSP.TSP import Graph

class TSPGWOSolver(GrayWolfOptimization):
    def __init__(self, graph, city_coords, population_size, dim, lower_bound, upper_bound, num_iterations=100):
        """Initialize the TSP GWO solver."""
        self.graph = graph
        self.city_coords = city_coords
        self.population_size = population_size
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_iterations = num_iterations
        
        # Initialize visualization attributes
        self.live_fig = None
        self.live_ax = None
        
        # Create TSP-specific problem instance
        problem = GWO_TSPProblem(
            graph=self.graph,
            city_coords=self.city_coords,
            population_size=self.population_size,
            dim=self.dim,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            fitness_function=None
        )
        super().__init__(problem)

    def setup_live_plot(self):
        """Initialize the live plotting window."""
        plt.ion()
        self.live_fig, self.live_ax = plt.subplots(figsize=(10, 10))
        self.live_ax.grid(True)
        self.live_ax.set_title('TSP Route (Live) - GWO Solver')
        plt.show()

    def update_live_plot(self, best_path, iteration, best_fitness):
        """Update the live plot with current best solution."""
        if best_path is None:
            return
        
        if self.live_ax is None:
            self.setup_live_plot()

        self.live_ax.clear()
        
        # Plot cities with annotations
        for i, (x, y) in enumerate(self.city_coords, 1):
            self.live_ax.plot(x, y, 'ro', markersize=10)
            self.live_ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')

        # Plot path with directional arrows
        path = np.array(best_path).astype(int)
        n = len(path)
        for i in range(n):
            city1 = path[i]
            city2 = path[(i + 1) % n]
            x1, y1 = self.city_coords[city1 - 1]
            x2, y2 = self.city_coords[city2 - 1]
            dx = x2 - x1
            dy = y2 - y1
            self.live_ax.arrow(x1, y1, dx, dy,
                             head_width=0.3,
                             head_length=0.3,
                             fc='blue',
                             ec='blue',
                             alpha=0.6)

        # Highlight starting city
        start_x, start_y = self.city_coords[0]  # Always start from city 1
        self.live_ax.plot(start_x, start_y, 'go', markersize=15, alpha=0.5, label='Start')
        
        self.live_ax.set_title(f'Generation {iteration}\nCurrent Distance: {best_fitness:.2f}')
        self.live_ax.axis('equal')
        self.live_ax.grid(True)
        self.live_ax.legend()

        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        plt.pause(0.01)

    def gwo_crossover(self, leader_pos, current_pos, crossover_rate=0.8):
        """TSP-specific crossover operation preserving route validity."""
        if np.random.random() > crossover_rate:
            return leader_pos.copy()

        # Always keep city 1 at the start
        n = len(leader_pos)
        child = np.array([1] + [-1] * (n-1))
        
        # Choose section after city 1
        start = 1 + np.random.randint(0, n-2)
        end = 1 + np.random.randint(start, n-1)
        
        # Copy section from leader
        child[start:end+1] = leader_pos[start:end+1]
        
        # Fill remaining positions from current wolf position
        remaining = [x for x in current_pos if x not in child[start:end+1] and x != 1]
        # Fill positions after end
        for i in range(end+1, n):
            child[i] = remaining.pop(0)
        # Fill positions between city 1 and start
        for i in range(1, start):
            child[i] = remaining.pop(0)
            
        return child

    def gwo_mutation(self, position, mutation_rate=0.1):
        """TSP-specific mutation operations."""
        if np.random.random() > mutation_rate:
            return position

        mutated = position.copy()
        mutation_type = np.random.choice(['swap', 'reverse', 'insert'], p=[0.4, 0.4, 0.2])
        
        if mutation_type == 'swap':
            # Swap two random cities (excluding city 1)
            i, j = np.random.choice(range(1, len(position)), 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            
        elif mutation_type == 'reverse':
            # Reverse a subsection (excluding city 1)
            i, j = sorted(np.random.choice(range(1, len(position)), 2, replace=False))
            mutated[i:j+1] = mutated[i:j+1][::-1]
            
        else:  # insert
            # Take a city and insert it at a random position (excluding city 1)
            i, j = np.random.choice(range(1, len(position)), 2, replace=False)
            city = mutated[i]
            mutated = np.delete(mutated, i)
            mutated = np.insert(mutated, j, city)

        return mutated

    def select_leader(self, population, fitness_values, exclude_indices=None, tournament_size=3):
        """Tournament selection for choosing leaders."""
        available_indices = np.arange(len(population))
        if exclude_indices is not None:
            mask = np.ones(len(population), dtype=bool)
            mask[exclude_indices] = False
            available_indices = available_indices[mask]
            
        if len(available_indices) < tournament_size:
            tournament_size = len(available_indices)
            
        tournament_indices = np.random.choice(available_indices, size=tournament_size, replace=False)
        tournament_fitness = fitness_values[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return winner_idx, population[winner_idx].copy()

    def update_positions(self, alpha_pos, beta_pos, delta_pos, a, iteration, max_iter):
        """Update positions using TSP-specific operators."""
        new_population = []
        progress_ratio = iteration / max_iter
        exploration_rate = 1 - progress_ratio
        
        for i in range(self.population_size):
            current_pos = self.problem.population[i]
            
            # Adaptive parameters
            crossover_rate = 0.8 - 0.4 * progress_ratio  # Decrease over time
            mutation_rate = 0.1 + 0.2 * exploration_rate  # Higher early on
            
            # Select leader based on progress and random factor
            r = np.random.random()
            if r < progress_ratio:  # Exploitation phase
                leader_pos = alpha_pos
            elif r < 0.6 + 0.2 * progress_ratio:  # Mixed phase
                leader_pos = beta_pos
            else:  # Exploration phase
                leader_pos = delta_pos
            
            # Apply crossover
            new_pos = self.gwo_crossover(leader_pos, current_pos, crossover_rate)
            
            # Apply mutation
            new_pos = self.gwo_mutation(new_pos, mutation_rate)
            
            # Local improvement with 2-opt when near convergence
            if progress_ratio > 0.8 and np.random.random() < 0.1:
                best_distance = self.problem.fitness(new_pos)
                improved = True
                while improved:
                    improved = False
                    for i in range(1, len(new_pos) - 2):
                        for j in range(i + 1, len(new_pos)):
                            new_route = new_pos.copy()
                            new_route[i:j] = new_route[j-1:i-1:-1]
                            new_distance = self.problem.fitness(new_route)
                            if new_distance < best_distance:
                                new_pos = new_route
                                best_distance = new_distance
                                improved = True
                                break
                        if improved:
                            break
            
            new_population.append(new_pos)
        
        self.problem.population = np.array(new_population)

    def optimize(self, max_iter=None):
        """Execute the GWO optimization process."""
        if max_iter is None:
            max_iter = self.num_iterations
            
        print(f"[GWO TSP] Starting optimization for {max_iter} iterations")
        self.setup_live_plot()
        
        best_fitness = float('inf')
        best_path = None
        stagnation_count = 0
        
        # Initialize population with city 1 fixed at start
        initial_population = []
        for _ in range(self.population_size):
            # Create random permutation starting with city 1
            perm = np.arange(2, self.dim + 1)  # Cities 2 to n
            np.random.shuffle(perm)
            perm = np.insert(perm, 0, 1)  # Add city 1 at start
            initial_population.append(perm)
        
        self.problem.population = np.array(initial_population)
        population = self.problem.population  # Get reference to population
        
        # Track convergence
        best_fitness_history = []
        iterations_without_improvement = 0
        last_improvement = 0
        
        for iter_num in range(max_iter):
            # Evaluate fitness
            fitness = np.array([self.problem.fitness(wolf) for wolf in population])
            
            # Select leaders using tournament selection
            exclude_indices = []
            
            # Select alpha
            alpha_idx, alpha_pos = self.select_leader(population, fitness)
            exclude_indices.append(alpha_idx)
            
            # Select beta
            beta_idx, beta_pos = self.select_leader(population, fitness, exclude_indices)
            exclude_indices.append(beta_idx)
            
            # Select delta
            delta_idx, delta_pos = self.select_leader(population, fitness, exclude_indices)
            
            # Update best solution if improved
            current_best_fitness = fitness[alpha_idx]
            if current_best_fitness < best_fitness:
                improvement = best_fitness - current_best_fitness
                best_fitness = current_best_fitness
                best_path = alpha_pos.copy()
                iterations_without_improvement = 0
                last_improvement = iter_num
                print(f"\nNew best distance: {best_fitness:.2f} (Improved by: {improvement:.2f})")
            else:
                iterations_without_improvement += 1
            
            # Strong perturbation if stuck
            if iterations_without_improvement > 20:
                print("\nStagnation detected, applying perturbation...")
                perturb_size = self.population_size // 4
                for idx in range(perturb_size):
                    # Create new solutions with city 1 fixed
                    perm = np.arange(2, self.dim + 1)
                    np.random.shuffle(perm)
                    perm = np.insert(perm, 0, 1)
                    population[-(idx+1)] = perm
                iterations_without_improvement = 0
            
            best_fitness_history.append(best_fitness)
            
            # Update positions
            a = 2 * (1 - iter_num / max_iter)
            self.update_positions(alpha_pos, beta_pos, delta_pos, a, iter_num, max_iter)
            population = self.problem.population  # Update reference to population
            
            # Update visualization
            if iter_num % 5 == 0:
                self.update_live_plot(best_path, iter_num + 1, best_fitness)
                print(f"Iteration {iter_num + 1}/{max_iter}, Best Distance: {best_fitness:.2f}")
            #show final result visualization for 10 seconds
            if iter_num == max_iter - 1:
                time.sleep(10)
                self.update_live_plot(best_path, max_iter, best_fitness)
                break
        # Final visualization
        self.update_live_plot(best_path, max_iter, best_fitness)
        
        # Plot convergence history
        plt.figure()
        plt.plot(best_fitness_history)
        plt.title('Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()
        
        print(f"\nOptimization complete. Best fitness: {best_fitness:.2f}")
        return best_path

if __name__ == '__main__':
    # Example usage with circular city layout
    num_cities = 20
    city_coords = []
    
    # Create cities in a circle with random displacement
    for i in range(num_cities):
        angle = 2 * np.pi * i / num_cities
        r = 5 + np.random.random()
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        city_coords.append((x, y))
    
    # Calculate distance matrix
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x1, y1 = city_coords[i]
                x2, y2 = city_coords[j]
                distance_matrix[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    graph = Graph(distance_matrix)
    
    # Configure solver
    population_size = 17
    dim = len(graph.get_vertices())
    lower_bound = 1
    upper_bound = num_cities
    num_iterations = 90
    
    # Create and run solver
    solver = TSPGWOSolver(graph, city_coords, population_size, dim, 
                         lower_bound, upper_bound, num_iterations)
    best_solution_path = solver.optimize()
    print("Best solution path (city indices 1-based):", best_solution_path)
