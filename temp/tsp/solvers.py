"""
TSP Solvers for the RL Orchestrator.

Provides exploration and exploitation solvers for permutation-based TSP:
- TSPExplorer: Diversity-maintaining exploration using edge-recombination + random restarts
- TSPExploiter: 2-opt local search for tour refinement
"""

import numpy as np
from typing import List, Optional
from temp.core.base import SearchAlgorithm, Solution


class TSPExplorer(SearchAlgorithm):
    """
    TSP Explorer using Order Crossover and high mutation for diversity.
    
    Maintains population diversity through:
    - Order Crossover (OX) to combine tour segments
    - High mutation rate with 2-opt moves
    - Random injection to prevent premature convergence
    """
    phase = "exploration"

    def __init__(self, problem, population_size: int, 
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


class TSPExploiter(SearchAlgorithm):
    """
    TSP Exploiter using intensive 2-opt local search.
    
    Focuses population on best solutions and applies local search
    to refine tours toward local optima.
    """
    phase = "exploitation"

    def __init__(self, problem, population_size: int, 
                 elite_ratio: float = 0.3,
                 local_search_iterations: int = 10, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.elite_ratio = elite_ratio
        self.local_search_iterations = local_search_iterations
        self._num_cities = None

    def initialize(self):
        """Initialize population."""
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        """One step of exploitation: local search on elite, focus population."""
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)
        
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        n_elite = max(1, int(self.population_size * self.elite_ratio))
        
        new_population = []
        
        # Apply local search to elite solutions
        for idx in range(n_elite):
            tour = np.array(sorted_pop[idx].representation)
            improved_tour, improved_fit = self._two_opt_local_search(tour)
            
            new_sol = Solution(improved_tour.tolist(), self.problem)
            new_sol.fitness = improved_fit
            new_population.append(new_sol)
        
        # Fill rest of population from elite with small perturbations
        while len(new_population) < self.population_size:
            elite_idx = np.random.randint(0, n_elite)
            elite_tour = np.array(sorted_pop[elite_idx].representation)
            
            # Small perturbation: 1-3 swap moves
            perturbed = elite_tour.copy()
            n_swaps = np.random.randint(1, 4)
            for _ in range(n_swaps):
                i, j = np.random.choice(self._num_cities, 2, replace=False)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            new_sol = Solution(perturbed.tolist(), self.problem)
            new_population.append(new_sol)
        
        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1

    def _two_opt_local_search(self, tour: np.ndarray) -> tuple:
        """Apply 2-opt local search until no improvement."""
        current_tour = tour.copy()
        
        # Create temporary solution for evaluation
        temp_sol = Solution(current_tour.tolist(), self.problem)
        temp_sol.evaluate()
        current_fitness = temp_sol.fitness
        
        improved = True
        iterations = 0
        
        while improved and iterations < self.local_search_iterations:
            improved = False
            iterations += 1
            
            # Try random 2-opt moves
            for _ in range(self._num_cities):
                i, j = sorted(np.random.choice(self._num_cities, 2, replace=False))
                if j - i < 2:
                    continue
                
                # Try reversing segment
                new_tour = current_tour.copy()
                new_tour[i:j+1] = new_tour[i:j+1][::-1]
                
                temp_sol = Solution(new_tour.tolist(), self.problem)
                temp_sol.evaluate()
                
                if temp_sol.fitness < current_fitness:
                    current_tour = new_tour
                    current_fitness = temp_sol.fitness
                    improved = True
                    break
        
        return current_tour, current_fitness

    def ingest_population(self, seeds: List[Solution]):
        """Ingest population and immediately apply local search to best."""
        super().ingest_population(seeds)
        
        if not self.population:
            return
        
        self._num_cities = len(self.population[0].representation)
        
        # Apply light local search to top solutions
        sorted_pop = sorted(self.population, key=lambda s: s.fitness if s.fitness else float('inf'))
        n_improve = min(3, len(sorted_pop))
        
        for idx in range(n_improve):
            if sorted_pop[idx].fitness is not None:
                tour = np.array(sorted_pop[idx].representation)
                improved_tour, improved_fit = self._two_opt_local_search(tour)
                sorted_pop[idx].representation = improved_tour.tolist()
                sorted_pop[idx].fitness = improved_fit
        
        self.population = sorted_pop
        self.mark_best_dirty()
        self._update_best_solution()
