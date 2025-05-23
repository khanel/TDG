"""
IGWO-based TSP Solver.

This module provides an implementation of an Improved Gray Wolf Optimization (IGWO)
solver for the Traveling Salesman Problem (TSP). It adapts the IGWO algorithm for
discrete permutation-based optimization required by TSP.
"""

import numpy as np
import matplotlib.pyplot as plt
from Core.search_algorithm import SearchAlgorithm
from Core.problem import Solution

class TSPIGWOSolver(SearchAlgorithm):
    def __init__(self, problem, city_coords, population_size=30, num_iterations=100, mutation_rate=0.2, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.city_coords = city_coords
        self.num_iterations = num_iterations
        self.mutation_rate = mutation_rate
        self.live_fig = None
        self.live_ax = None
        self.pbest = []
        self.pbest_score = []
        self.alpha = None
        self.beta = None
        self.delta = None
        self.alpha_score = None
        self.beta_score = None
        self.delta_score = None

    def initialize(self):
        self.population = [self.problem.get_initial_solution() for _ in range(self.population_size)]
        for sol in self.population:
            sol.evaluate()
        self.pbest = [Solution(sol.representation.copy(), sol.problem) for sol in self.population]
        for sol in self.pbest:
            sol.evaluate()
        self.pbest_score = [sol.fitness for sol in self.pbest]
        self._update_leaders()
        self._update_best_solution()

    def step(self):
        # Discrete IGWO step for TSP
        new_population = []
        for i in range(self.population_size):
            current = self.population[i]
            # Select leaders
            alpha = self.alpha
            beta = self.beta
            # Crossover with alpha and beta
            child_repr = self.tsp_crossover(alpha.representation, beta.representation)
            # Mutation
            child_repr = self.tsp_mutation(child_repr)
            child = Solution(child_repr, self.problem)
            child.evaluate()
            # Update pbest
            if child.fitness < self.pbest_score[i]:
                self.pbest[i] = Solution(child.representation.copy(), child.problem)
                self.pbest[i].evaluate()
                self.pbest_score[i] = child.fitness
            new_population.append(child)
        self.population = new_population
        self._update_leaders()
        self._update_best_solution()
        self.iteration += 1

    def tsp_crossover(self, parent1, parent2):
        # Order Crossover (OX) for permutations
        n = len(parent1)
        start = 1 + np.random.randint(0, n - 2)
        end = 1 + np.random.randint(start, n - 1)
        child = np.ones(n, dtype=int) * -1
        child[0] = 1  # Always start at city 1
        child[start:end + 1] = parent1[start:end + 1]
        fill = [x for x in parent2 if x not in child[start:end + 1] and x != 1]
        idx = end + 1
        for x in fill:
            if idx >= n:
                idx = 1
            while child[idx] != -1:
                idx += 1
                if idx >= n:
                    idx = 1
            child[idx] = x
            idx += 1
        return child

    def tsp_mutation(self, individual):
        if np.random.rand() > self.mutation_rate:
            return individual
        n = len(individual)
        pos1, pos2 = np.random.choice(range(1, n), size=2, replace=False)
        mutated = individual.copy()
        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
        return mutated

    def _update_leaders(self):
        # Find best three individuals
        fitnesses = [sol.fitness for sol in self.population]
        sorted_indices = np.argsort(fitnesses)
        self.alpha = self.population[sorted_indices[0]]
        self.beta = self.population[sorted_indices[1]]
        self.delta = self.population[sorted_indices[2]]
        self.alpha_score = fitnesses[sorted_indices[0]]
        self.beta_score = fitnesses[sorted_indices[1]]
        self.delta_score = fitnesses[sorted_indices[2]]

    def optimize(self, max_iter=None):
        if max_iter is None:
            max_iter = self.num_iterations
        print(f"[IGWO TSP] Starting optimization for {max_iter} iterations")
        self.setup_live_plot()
        self.initialize()
        best_fitness_history = []
        best_path = None
        for iter_num in range(max_iter):
            self.step()
            best_fitness_history.append(self.alpha_score)
            best_path = self.alpha.representation.copy()
            self.update_live_plot(best_path, iter_num, self.alpha_score)
            if iter_num % 10 == 0:
                print(f"Iteration {iter_num}: Best Distance = {self.alpha_score:.2f}")
        plt.figure()
        plt.plot(best_fitness_history)
        plt.title('IGWO-TSP Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.grid(True)
        plt.show()
        print(f"\nOptimization complete. Best distance: {self.alpha_score:.2f}")
        # Show the final best route and block until manually closed
        self.update_live_plot(best_path, max_iter - 1, self.alpha_score)
        plt.ioff()
        plt.show(block=True)
        return best_path

    def setup_live_plot(self):
        plt.ion()
        self.live_fig, self.live_ax = plt.subplots(figsize=(10, 10))
        self.live_ax.grid(True)
        self.live_ax.set_title('TSP Route (Live) - IGWO Solver')
        plt.show()

    def update_live_plot(self, best_path, iteration, best_fitness):
        if best_path is None:
            return
        if self.live_ax is None:
            self.setup_live_plot()
        self.live_ax.clear()
        for i, (x, y) in enumerate(self.city_coords, 1):
            self.live_ax.plot(x, y, 'ro', markersize=10)
            self.live_ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        path = np.array(best_path).astype(int)
        n = len(path)
        for i in range(n):
            city1 = path[i] - 1
            city2 = path[(i + 1) % n] - 1
            x1, y1 = self.city_coords[city1]
            x2, y2 = self.city_coords[city2]
            self.live_ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.18, head_length=0.3, fc='blue', ec='blue', alpha=0.6)
        start_x, start_y = self.city_coords[0]
        self.live_ax.plot(start_x, start_y, 'go', markersize=15, alpha=0.5, label='Start')
        self.live_ax.set_title(f'Generation {iteration}\nCurrent Distance: {best_fitness:.2f}')
        self.live_ax.axis('equal')
        self.live_ax.grid(True)
        self.live_ax.legend()
        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        plt.pause(0.01)

if __name__ == '__main__':
    # Example usage with circular city layout
    from TSP.TSP import Graph, TSPProblem
    num_cities = 17
    city_coords = []
    for i in range(num_cities):
        angle = 2 * np.pi * i / num_cities
        r = 5 + np.random.random()
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        city_coords.append((x, y))
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x1, y1 = city_coords[i]
                x2, y2 = city_coords[j]
                distance_matrix[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    graph = Graph(distance_matrix)
    problem = TSPProblem(graph)
    population_size = 17
    num_iterations = 900
    solver = TSPIGWOSolver(problem, city_coords, population_size, num_iterations)
    best_solution_path = solver.optimize()
    print("Best solution path (city indices 1-based):", best_solution_path)

    def update_position(self, t):
        # Discrete update using TSP-specific crossover and mutation
        new_population = []
        for i in range(self.population_size):
            current = self.population[i]
            # Select leaders
            alpha = self.alpha
            beta = self.beta
            # Crossover with alpha and beta
            child = self.tsp_crossover(alpha, beta)
            # Mutation
            child = self.tsp_mutation(child)
            new_population.append(child)
        self.population = np.array(new_population)
        self.fitness = np.array([self.tsp_fitness(ind) for ind in self.population])
        # Update personal bests
        for i in range(self.population_size):
            if self.fitness[i] < self.pbest_score[i]:
                self.pbest_score[i] = self.fitness[i]
                self.pbest_pos[i] = self.population[i].copy()

    def tsp_crossover(self, parent1, parent2):
        # Order Crossover (OX) for permutations
        n = len(parent1)
        start = 1 + np.random.randint(0, n - 2)
        end = 1 + np.random.randint(start, n - 1)
        child = np.ones(n, dtype=int) * -1
        child[0] = 1  # Always start at city 1
        child[start:end + 1] = parent1[start:end + 1]
        fill = [x for x in parent2 if x not in child[start:end + 1] and x != 1]
        idx = end + 1
        for x in fill:
            if idx >= n:
                idx = 1
            while child[idx] != -1:
                idx += 1
                if idx >= n:
                    idx = 1
            child[idx] = x
            idx += 1
        return child

    def tsp_mutation(self, individual, mutation_rate=0.2):
        if np.random.rand() > mutation_rate:
            return individual
        n = len(individual)
        pos1, pos2 = np.random.choice(range(1, n), size=2, replace=False)
        mutated = individual.copy()
        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
        return mutated

    def update_leaders(self):
        # Find best three individuals
        sorted_indices = np.argsort(self.fitness)
        self.alpha = self.population[sorted_indices[0]].copy()
        self.beta = self.population[sorted_indices[1]].copy()
        self.delta = self.population[sorted_indices[2]].copy()
        self.alpha_score = self.fitness[sorted_indices[0]]
        self.beta_score = self.fitness[sorted_indices[1]]
        self.delta_score = self.fitness[sorted_indices[2]]

    def optimize(self, max_iter=None):
        if max_iter is None:
            max_iter = self.num_iterations
        print(f"[IGWO TSP] Starting optimization for {max_iter} iterations")
        self.setup_live_plot()
        self.initialize_population()
        self.update_leaders()
        best_fitness_history = []
        best_path = None
        for iter_num in range(max_iter):
            self.update_position(iter_num)
            self.update_leaders()
            best_fitness_history.append(self.alpha_score)
            best_path = self.alpha.copy()
            self.update_live_plot(best_path, iter_num, self.alpha_score)
            if iter_num % 10 == 0:
                print(f"Iteration {iter_num}: Best Distance = {self.alpha_score:.2f}")
        plt.figure()
        plt.plot(best_fitness_history)
        plt.title('IGWO-TSP Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.grid(True)
        plt.show()
        print(f"\nOptimization complete. Best distance: {self.alpha_score:.2f}")
        # Show the final best route and block until manually closed
        self.update_live_plot(best_path, max_iter - 1, self.alpha_score)
        plt.ioff()
        plt.show(block=True)
        return best_path

    def setup_live_plot(self):
        plt.ion()
        self.live_fig, self.live_ax = plt.subplots(figsize=(10, 10))
        self.live_ax.grid(True)
        self.live_ax.set_title('TSP Route (Live) - IGWO Solver')
        plt.show()

    def update_live_plot(self, best_path, iteration, best_fitness):
        if best_path is None:
            return
        if self.live_ax is None:
            self.setup_live_plot()
        self.live_ax.clear()
        for i, (x, y) in enumerate(self.city_coords, 1):
            self.live_ax.plot(x, y, 'ro', markersize=10)
            self.live_ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        path = np.array(best_path).astype(int)
        n = len(path)
        for i in range(n):
            city1 = path[i] - 1
            city2 = path[(i + 1) % n] - 1
            x1, y1 = self.city_coords[city1]
            x2, y2 = self.city_coords[city2]
            self.live_ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.18, head_length=0.3, fc='blue', ec='blue', alpha=0.6)
        start_x, start_y = self.city_coords[0]
        self.live_ax.plot(start_x, start_y, 'go', markersize=15, alpha=0.5, label='Start')
        self.live_ax.set_title(f'Generation {iteration}\nCurrent Distance: {best_fitness:.2f}')
        self.live_ax.axis('equal')
        self.live_ax.grid(True)
        self.live_ax.legend()
        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        plt.pause(0.01)

if __name__ == '__main__':
    # Example usage with circular city layout
    num_cities = 17
    city_coords = []
    for i in range(num_cities):
        angle = 2 * np.pi * i / num_cities
        r = 5 + np.random.random()
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        city_coords.append((x, y))
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x1, y1 = city_coords[i]
                x2, y2 = city_coords[j]
                distance_matrix[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    graph = Graph(distance_matrix)
    population_size = 17
    dim = len(graph.get_vertices())
    lower_bound = 1
    upper_bound = num_cities
    num_iterations = 900
    solver = TSPIGWOSolver(graph, city_coords, population_size, dim, lower_bound, upper_bound, num_iterations)
    best_solution_path = solver.optimize()
    print("Best solution path (city indices 1-based):", best_solution_path)
