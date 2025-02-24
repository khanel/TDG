#!/usr/bin/env python3
"""
GWO-based TSP Solver.

This module provides an implementation of a Gray Wolf Optimization (GWO)
solver for the Traveling Salesman Problem (TSP). It extends the base GWO
framework with TSP-specific operators.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from GWO.GWO import GrayWolfOptimization
from GWO.Problem import GWO_TSPProblem
from TSP.TSP import Graph

class TSPGWOSolver(GrayWolfOptimization):
    """
    TSP solver using Gray Wolf Optimization algorithm.
    Implements TSP-specific operations and visualization.
    """
    def __init__(self, graph, city_coords, population_size, dim, lower_bound, upper_bound, num_iterations=100):
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
        self.live_ax.set_title('TSP Route (Live)')
        plt.show()

    def update_live_plot(self, best_path, iteration, best_fitness):
        """Update the live plot with current best solution."""
        if best_path is None:
            return
            
        if self.live_ax is None:
            self.setup_live_plot()
            
        self.live_ax.clear()
        
        # Plot cities
        for i, (x, y) in enumerate(self.city_coords, 1):
            self.live_ax.plot(x, y, 'ro', markersize=10)
            self.live_ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot path with directional arrows
        path = np.array(best_path).astype(int)
        n = len(path)
        for i in range(n):
            city1, city2 = path[i], path[(i + 1) % n]
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
        start_x, start_y = self.city_coords[path[0] - 1]
        self.live_ax.plot(start_x, start_y, 'go', markersize=15, alpha=0.5, label='Start')
        
        self.live_ax.set_title(f'Generation {iteration}\nCurrent Distance: {best_fitness:.2f}')
        self.live_ax.axis('equal')
        self.live_ax.grid(True)
        self.live_ax.legend()

        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        plt.pause(0.01)

    def update_positions(self, alpha_pos, beta_pos, delta_pos, a, iteration, max_iter):
        """Update positions using TSP-specific operators."""
        new_population = []
        progress_ratio = iteration / max_iter
        exploration_rate = 1 - progress_ratio
        
        for i in range(self.population_size):
            current_pos = self.problem.population[i].copy()
            
            # Select leader based on progress
            r = np.random.random()
            if r < progress_ratio:
                leader_pos = alpha_pos
            elif r < 0.6 + 0.2 * progress_ratio:
                leader_pos = beta_pos
            else:
                leader_pos = delta_pos
                
            # Create new position using order crossover
            if np.random.random() < (0.8 - 0.4 * progress_ratio):
                # Order Crossover (OX)
                start = 1 + np.random.randint(0, self.dim - 2)
                end = 1 + np.random.randint(start, self.dim - 1)
                new_pos = np.ones(self.dim) * -1
                new_pos[0] = 1  # Keep city 1 as start
                
                # Copy segment from leader
                new_pos[start:end+1] = leader_pos[start:end+1]
                
                # Fill remaining positions from current
                remaining = [x for x in current_pos if x not in new_pos]
                fill_idx = 1
                for val in remaining:
                    while fill_idx < self.dim and new_pos[fill_idx] != -1:
                        fill_idx += 1
                    if fill_idx < self.dim:
                        new_pos[fill_idx] = val
            else:
                # Mutation
                new_pos = leader_pos.copy()
                mutation_strength = exploration_rate * 0.4
                num_swaps = max(1, int(mutation_strength * (self.dim - 1)))
                
                for _ in range(num_swaps):
                    i, j = np.random.choice(range(1, self.dim), 2, replace=False)
                    new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
            
            new_population.append(new_pos)
            
        self.problem.population = np.array(new_population)

    def optimize(self, max_iter=None):
        """Execute the optimization process."""
        if max_iter is None:
            max_iter = self.num_iterations
            
        print(f"[GWO TSP] Starting optimization for {max_iter} iterations")
        best_fitness = float('inf')
        best_path = None
        
        # Initialize population with city 1 fixed at start
        initial_population = []
        for _ in range(self.population_size):
            perm = np.arange(2, self.dim + 1)
            np.random.shuffle(perm)
            perm = np.insert(perm, 0, 1)
            initial_population.append(perm)
        
        self.problem.population = np.array(initial_population)
        population = self.problem.population
        
        # Track progress
        best_fitness_history = []
        iterations_without_improvement = 0
        
        for iter_num in range(max_iter):
            # Evaluate fitness
            fitness = np.array([self.problem.fitness(wolf) for wolf in population])
            
            # Update hierarchy
            sorted_indices = np.argsort(fitness)
            alpha_idx = sorted_indices[0]
            alpha_pos = population[alpha_idx].copy()
            beta_pos = population[sorted_indices[1]].copy()
            delta_pos = population[sorted_indices[2]].copy()
            
            # Update best solution
            current_best_fitness = fitness[alpha_idx]
            if current_best_fitness < best_fitness:
                improvement = best_fitness - current_best_fitness
                best_fitness = current_best_fitness
                best_path = alpha_pos.copy()
                iterations_without_improvement = 0
                print(f"\nNew best distance: {best_fitness:.2f} (Improved by: {improvement:.2f})")
            else:
                iterations_without_improvement += 1
            
            best_fitness_history.append(best_fitness)
            
            # Apply perturbation if stuck
            if iterations_without_improvement > 20:
                print("\nStagnation detected, applying perturbation...")
                perturb_size = self.population_size // 4
                for idx in range(perturb_size):
                    perm = np.arange(2, self.dim + 1)
                    np.random.shuffle(perm)
                    perm = np.insert(perm, 0, 1)
                    population[-(idx+1)] = perm
                iterations_without_improvement = 0
            
            # Update positions
            a = 2 * (1 - iter_num / max_iter)
            self.update_positions(alpha_pos, beta_pos, delta_pos, a, iter_num, max_iter)
            population = self.problem.population
            
            # Visualization
            if iter_num % 5 == 0:
                self.update_live_plot(best_path, iter_num + 1, best_fitness)
                print(f"Iteration {iter_num + 1}/{max_iter}, Best Distance: {best_fitness:.2f}")
        
        # Final updates
        self.update_live_plot(best_path, max_iter, best_fitness)
        
        # Plot convergence
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
    population_size = 100
    dim = len(graph.get_vertices())
    lower_bound = 1
    upper_bound = num_cities
    num_iterations = 1000
    
    # Create and run solver
    solver = TSPGWOSolver(graph, city_coords, population_size, dim, 
                         lower_bound, upper_bound, num_iterations)
    best_solution_path = solver.optimize()
    print("Best solution path (city indices 1-based):", best_solution_path)