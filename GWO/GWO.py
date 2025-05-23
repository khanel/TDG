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
        self._update_best_solution()
        self.iteration += 1

    def _update_position(self, wolf, leader, a):
        r1 = np.random.rand(*np.shape(wolf.representation))
        r2 = np.random.rand(*np.shape(wolf.representation))
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * leader.representation - wolf.representation)
        X = leader.representation - A * D
        return X