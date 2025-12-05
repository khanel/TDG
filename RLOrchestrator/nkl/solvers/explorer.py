"""
MAP-Elites Quality-Diversity algorithm for EXPLORATION.
"""

from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


class Elite:
    """Represents an elite solution in the archive."""
    x: np.ndarray
    fitness: float
    bd: np.ndarray  # Behavior descriptor


class NKLMapElitesExplorer(SearchAlgorithm):
    """
    MAP-Elites Quality-Diversity algorithm for EXPLORATION.
    
    Maintains an archive of diverse, high-quality solutions across
    a behavioral space. Perfect for exploration as it:
    - Explicitly maintains diversity through behavioral descriptors
    - Keeps the best solution in each behavioral niche
    - Natural exploration pressure to fill empty niches
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int, 
                 n_bins: int = 10, mutation_rate: float = 0.1,
                 batch_size: int = 32, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.n_bins = n_bins
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self._dimension = None
        
        # Archive: maps (bin_x, bin_y) -> Elite
        self.archive: Dict[Tuple[int, int], Elite] = {}
        
    def _compute_behavior_descriptor(self, x: np.ndarray) -> np.ndarray:
        """
        Compute 2D behavior descriptor for binary solution.
        Uses: (proportion of 1s in first half, proportion of 1s in second half)
        """
        mid = len(x) // 2
        bd1 = np.mean(x[:mid])  # Density in first half
        bd2 = np.mean(x[mid:])  # Density in second half
        return np.array([bd1, bd2])
    
    def _bd_to_key(self, bd: np.ndarray) -> Tuple[int, int]:
        """Convert behavior descriptor to archive key."""
        bin_x = int(np.clip(bd[0] * self.n_bins, 0, self.n_bins - 1))
        bin_y = int(np.clip(bd[1] * self.n_bins, 0, self.n_bins - 1))
        return (bin_x, bin_y)
    
    def _add_to_archive(self, x: np.ndarray, fitness: float) -> bool:
        """Add solution to archive if it improves the cell. Returns True if added."""
        bd = self._compute_behavior_descriptor(x)
        key = self._bd_to_key(bd)
        
        current = self.archive.get(key)
        if current is None or fitness < current.fitness:  # Minimization
            self.archive[key] = Elite(x.copy(), fitness, bd)
            return True
        return False

    def initialize(self):
        """Initialize population and archive."""
        super().initialize()
        
        if not self.population:
            return
            
        self._dimension = len(self.population[0].representation)
        
        # Add initial population to archive
        for sol in self.population:
            self._add_to_archive(np.asarray(sol.representation), sol.fitness)
        
        self._update_best_solution()

    def step(self):
        """One step of MAP-Elites: select parents, mutate, add to archive."""
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        if not self.archive:
            self.initialize()
            return
        
        # Select parents from archive uniformly
        archive_elites = list(self.archive.values())
        
        new_solutions = []
        for _ in range(self.batch_size):
            # Select random parent from archive
            parent = archive_elites[np.random.randint(len(archive_elites))]
            parent_x = parent.x.copy()
            
            # Mutation: flip bits with probability mutation_rate
            mutation_mask = np.random.rand(self._dimension) < self.mutation_rate
            child_x = parent_x.copy()
            child_x[mutation_mask] = 1 - child_x[mutation_mask]
            
            # Evaluate and add to archive
            child_sol = Solution(child_x.astype(int), self.problem)
            child_sol.evaluate()
            
            self._add_to_archive(child_x, child_sol.fitness)
            new_solutions.append(child_sol)
        
        # Update population from archive (sample diverse solutions)
        archive_solutions = []
        for elite in self.archive.values():
            sol = Solution(elite.x.astype(int), self.problem)
            sol.fitness = elite.fitness
            archive_solutions.append(sol)
        
        # Keep population_size solutions, preferring diversity
        if len(archive_solutions) >= self.population_size:
            self.population = archive_solutions[:self.population_size]
        else:
            self.population = archive_solutions + new_solutions[:self.population_size - len(archive_solutions)]
        
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1
    
    def get_archive_coverage(self) -> float:
        """Return fraction of archive cells filled."""
        total_cells = self.n_bins * self.n_bins
        return len(self.archive) / total_cells
    
    def ingest_population(self, seeds: List[Solution]):
        """Ingest population and add to archive."""
        super().ingest_population(seeds)
        for sol in self.population:
            if sol.fitness is not None:
                self._add_to_archive(np.asarray(sol.representation), sol.fitness)