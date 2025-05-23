import numpy as np
"""
Generic Gray Wolf Optimization (GWO) Module.

This module provides a base for a generic and reusable GWO algorithm.
It includes an abstract base class that defines the generic interface.
"""

import abc

from Core.search_algorithm import SearchAlgorithm
from Core.problem import Solution

class GrayWolfOptimization(SearchAlgorithm):
    def __init__(self, problem, population_size, max_iterations=100, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.max_iterations = max_iterations
        self.iteration = 0

    def initialize(self):
        self.population = self.problem.get_initial_population(self.population_size)
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()

    def step(self):
        # Check problem type
        problem_info = self.problem.get_problem_info()
        problem_type = problem_info.get('problem_type', 'continuous')
        
        if problem_type == 'discrete':
            # For discrete problems like TSP, use a discrete adaptation of GWO
            self._discrete_step()
        else:
            # For continuous problems, use the standard GWO
            self._continuous_step()
            
        self._update_best_solution()
        self.iteration += 1
        
    def _discrete_step(self):
        """
        Adaptation of GWO for discrete problems like TSP.
        For TSP, we use a position-based crossover method.
        """
        # Sort wolves by fitness
        self.population.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        
        # Alpha, beta, and delta are the three best solutions
        alpha, beta, delta = self.population[0], self.population[1], self.population[2]
        
        new_population = []
        # Apply a discrete updating strategy for each wolf
        for wolf in self.population:
            # Create a new solution by applying crossover between wolf and alpha/beta/delta
            # with decreasing probability of following the leader as iteration increases
            a = 2 - self.iteration * (2 / self.max_iterations)
            r = np.random.rand()
            
            # Simple discrete adaptation: follow the best solutions with probability based on 'a'
            if r < a:  # Higher probability in early iterations
                # Apply a simple crossover with one of the leaders
                leader = np.random.choice([alpha, beta, delta])
                # Create a new solution by copying a segment from the leader
                new_repr = self._discrete_crossover(wolf.representation, leader.representation)
            else:
                # Apply a random permutation (exploration)
                new_repr = wolf.representation.copy()
                # Swap two random cities (excluding city 1 which is fixed at the start)
                if len(new_repr) > 3:  # Need at least 3 cities to swap (since city 1 is fixed)
                    i, j = np.random.choice(range(1, len(new_repr)), size=2, replace=False)
                    new_repr[i], new_repr[j] = new_repr[j], new_repr[i]
            
            # Create and evaluate new solution
            new_sol = Solution(new_repr, self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)
            
        self.population = new_population
    
    def _discrete_crossover(self, wolf_repr, leader_repr):
        """Simple crossover for discrete problems like TSP."""
        # Create a new solution that follows the leader partially
        n = len(wolf_repr)
        # Start with city 1 (fixed)
        new_repr = [1]
        
        # Copy a random segment from the leader (maintaining relative order)
        segment_length = np.random.randint(1, n // 2)
        start_pos = np.random.randint(1, n - segment_length)
        segment = leader_repr[start_pos:start_pos + segment_length]
        
        # Add cities from the segment that aren't already in new_repr
        for city in segment:
            if city not in new_repr:
                new_repr.append(city)
        
        # Add remaining cities from wolf in their original order
        for city in wolf_repr:
            if city not in new_repr:
                new_repr.append(city)
                
        return new_repr
        
    def _continuous_step(self):
        """Original GWO algorithm for continuous problems."""
        # Evaluate fitness and find alpha, beta, delta
        fitness = np.array([sol.fitness if sol.fitness is not None else sol.evaluate() for sol in self.population])
        idx = np.argsort(fitness)
        alpha, beta, delta = self.population[idx[0]], self.population[idx[1]], self.population[idx[2]]

        # Linearly decrease 'a' from 2 to 0
        a = 2 - self.iteration * (2 / self.max_iterations)

        new_population = []
        for wolf in self.population:
            X1 = self._update_position(wolf, alpha, a)
            X2 = self._update_position(wolf, beta, a)
            X3 = self._update_position(wolf, delta, a)
            new_repr = (np.array(X1) + np.array(X2) + np.array(X3)) / 3
            # Boundary handling
            info = self.problem.get_problem_info()
            lb = np.array(info.get('lower_bounds', -np.inf))
            ub = np.array(info.get('upper_bounds', np.inf))
            new_repr = np.clip(new_repr, lb, ub)
            new_sol = Solution(new_repr, self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)
        self.population = new_population

    def _update_position(self, wolf, leader, a):
        r1 = np.random.rand(*np.shape(wolf.representation))
        r2 = np.random.rand(*np.shape(wolf.representation))
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * leader.representation - wolf.representation)
        X = leader.representation - A * D
        return X