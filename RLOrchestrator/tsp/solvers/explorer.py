"""
TSP Explorer using Order Crossover and high mutation for diversity.
"""

from __future__ import annotations

from typing import List

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


class TSPMapElitesExplorer(SearchAlgorithm):
    """
    TSP Explorer using Order Crossover and high mutation for diversity.
    
    Maintains population diversity through:
    - Order Crossover (OX) to combine tour segments
    - High mutation rate with 2-opt moves
    - Random injection to prevent premature convergence
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int, 
                 mutation_rate: float = 0.3, 
                 random_injection_rate: float = 0.15, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.random_injection_rate = random_injection_rate
        self._num_cities = None

    def initialize(self):
        """Initialize with random tours."""
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        """One step of exploration: crossover, mutation, selection."""
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)
        
        new_population = []
        
        # Keep best solution (light elitism)
        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        new_population.append(sorted_pop[0].copy(preserve_id=False))
        
        while len(new_population) < self.population_size:
            if np.random.rand() < self.random_injection_rate:
                # Random injection for diversity
                new_tour = np.random.permutation(self._num_cities).tolist()
            else:
                # Tournament selection
                idx1, idx2 = np.random.choice(len(self.population), 2, replace=False)
                parent1 = self.population[idx1] if self.population[idx1].fitness < self.population[idx2].fitness else self.population[idx2]
                
                idx3, idx4 = np.random.choice(len(self.population), 2, replace=False)
                parent2 = self.population[idx3] if self.population[idx3].fitness < self.population[idx4].fitness else self.population[idx4]
                
                # Order Crossover (OX)
                new_tour = self._order_crossover(
                    np.array(parent1.representation),
                    np.array(parent2.representation)
                )
                
                # Mutation: multiple 2-opt moves
                if np.random.rand() < self.mutation_rate:
                    n_mutations = np.random.randint(1, 4)
                    for _ in range(n_mutations):
                        new_tour = self._two_opt_move(new_tour)
            
            new_sol = Solution(list(new_tour), self.problem)
            new_population.append(new_sol)
        
        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1

    def _order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Order Crossover (OX) for permutations."""
        n = len(parent1)
        
        # Select crossover segment
        start, end = sorted(np.random.choice(n, 2, replace=False))
        
        # Child starts with segment from parent1
        child = np.full(n, -1)
        child[start:end+1] = parent1[start:end+1]
        
        # Fill remaining from parent2 in order
        segment_set = set(child[start:end+1])
        p2_remaining = [x for x in parent2 if x not in segment_set]
        
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = p2_remaining[idx]
                idx += 1
        
        return child

    def _two_opt_move(self, tour: np.ndarray) -> np.ndarray:
        """Apply a single 2-opt move (reverse a segment)."""
        n = len(tour)
        i, j = sorted(np.random.choice(n, 2, replace=False))
        new_tour = tour.copy()
        new_tour[i:j+1] = new_tour[i:j+1][::-1]
        return new_tour
