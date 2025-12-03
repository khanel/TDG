
"""
This file contains self-contained implementations of the search algorithms
(solvers) used in the training process. Includes:
- MAPElitesExplorer: Quality-Diversity exploration using MAP-Elites
- BinaryPSO: Particle swarm exploitation
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

# We need the base classes from core.py
from temp.core.base import SearchAlgorithm, Solution


# --- MAP-Elites for Exploration ---

@dataclass
class Elite:
    """Represents an elite solution in the archive."""
    x: np.ndarray
    fitness: float
    bd: np.ndarray  # Behavior descriptor


class MAPElitesExplorer(SearchAlgorithm):
    """
    MAP-Elites Quality-Diversity algorithm for EXPLORATION.
    
    Maintains an archive of diverse, high-quality solutions across
    a behavioral space. Perfect for exploration as it:
    - Explicitly maintains diversity through behavioral descriptors
    - Keeps the best solution in each behavioral niche
    - Natural exploration pressure to fill empty niches
    """
    phase = "exploration"

    def __init__(self, problem, population_size, 
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


# --- Simple Diversity Explorer ---

class DiversityExplorer(SearchAlgorithm):
    """
    Fast diversity-maintaining explorer using random restarts and mutation.
    
    Key properties:
    - Maintains population diversity through high mutation
    - Uses tournament selection to keep some fitness pressure
    - Random injection to prevent convergence
    """
    phase = "exploration"

    def __init__(self, problem, population_size, mutation_rate: float = 0.15,
                 random_injection_rate: float = 0.2, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.random_injection_rate = random_injection_rate
        self._dimension = None

    def initialize(self):
        """Initialize with random population."""
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        """One step of diversity exploration."""
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        new_population = []
        
        # Keep best solution (light elitism)
        best = min(self.population, key=lambda s: s.fitness)
        new_population.append(best.copy(preserve_id=False))
        
        for _ in range(self.population_size - 1):
            if np.random.rand() < self.random_injection_rate:
                # Random injection for diversity
                new_x = np.random.randint(0, 2, self._dimension)
            else:
                # Tournament selection + mutation
                idx1, idx2 = np.random.choice(len(self.population), 2, replace=False)
                parent = self.population[idx1] if self.population[idx1].fitness < self.population[idx2].fitness else self.population[idx2]
                parent_x = np.asarray(parent.representation)
                
                # Heavy mutation for exploration
                mutation_mask = np.random.rand(self._dimension) < self.mutation_rate
                new_x = parent_x.copy()
                new_x[mutation_mask] = 1 - new_x[mutation_mask]
            
            new_sol = Solution(new_x.astype(int), self.problem)
            new_population.append(new_sol)
        
        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1


# --- Local Search Exploiter ---

class LocalSearchExploiter(SearchAlgorithm):
    """
    Local search (1-bit flip hill climbing) for pure exploitation.
    
    This is a classic exploitation strategy that refines solutions
    by making small local improvements. Also focuses population on best.
    """
    phase = "exploitation"

    def __init__(self, problem, population_size, elite_ratio: float = 0.2, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self._dimension = None
        self.elite_ratio = elite_ratio

    def initialize(self):
        """Initialize population."""
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        """One step of local search - try 1-bit flips on all solutions."""
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        # Sort by fitness (best first)
        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        
        # Elite solutions get local search
        n_elite = max(1, int(self.population_size * self.elite_ratio))
        
        new_population = []
        
        for idx, sol in enumerate(sorted_pop):
            current_x = np.asarray(sol.representation)
            current_fit = sol.fitness
            
            if idx < n_elite:
                # Elite - do intensive local search
                best_neighbor_x = current_x.copy()
                best_neighbor_fit = current_fit
                
                # Try more positions for elite
                positions_to_try = np.random.choice(self._dimension, min(15, self._dimension), replace=False)
                
                for pos in positions_to_try:
                    neighbor_x = current_x.copy()
                    neighbor_x[pos] = 1 - neighbor_x[pos]
                    neighbor_sol = Solution(neighbor_x.astype(int), self.problem)
                    neighbor_sol.evaluate()
                    
                    if neighbor_sol.fitness < best_neighbor_fit:
                        best_neighbor_x = neighbor_x
                        best_neighbor_fit = neighbor_sol.fitness
                
                new_sol = Solution(best_neighbor_x.astype(int), self.problem)
                new_sol.fitness = best_neighbor_fit
            else:
                # Non-elite - copy from elite with small mutation (focus population)
                elite_idx = np.random.randint(0, n_elite)
                elite_x = np.asarray(sorted_pop[elite_idx].representation).copy()
                
                # Small mutation (1-3 bits)
                n_flip = np.random.randint(1, 4)
                flip_pos = np.random.choice(self._dimension, n_flip, replace=False)
                elite_x[flip_pos] = 1 - elite_x[flip_pos]
                
                new_sol = Solution(elite_x.astype(int), self.problem)
            
            new_population.append(new_sol)
        
        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1


# --- Binary PSO for Exploitation ---

class BinaryPSO(SearchAlgorithm):
    """
    Binary Particle Swarm Optimization for PURE EXPLOITATION.
    
    Uses deterministic bit-flip based on velocity sign for strong convergence.
    """
    phase = "exploitation"

    def __init__(self, problem, population_size, omega: float = 0.7,
                 c1: float = 1.5, c2: float = 2.0, vmax: float = 4.0, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.vmax = vmax
        
        self.velocities = None
        self.personal_bests = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = float('inf')
        self._dimension = None

    def initialize(self):
        """Initialize the swarm."""
        super().initialize()
        
        if not self.population:
            return
        
        self._dimension = len(self.population[0].representation)
        
        # Vectorized initialization
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self._dimension))
        
        self.personal_bests = np.array([np.asarray(sol.representation) for sol in self.population])
        self.personal_best_fitness = np.array([sol.fitness for sol in self.population])
        
        self._update_global_best()

    def _update_global_best(self):
        """Update global best."""
        if self.personal_best_fitness is not None:
            best_idx = np.argmin(self.personal_best_fitness)
            if self.personal_best_fitness[best_idx] < self.global_best_fitness:
                self.global_best = self.personal_bests[best_idx].copy()
                self.global_best_fitness = self.personal_best_fitness[best_idx]
        
        if self.global_best is not None:
            self.best_solution = Solution(self.global_best.astype(int), self.problem)
            self.best_solution.fitness = self.global_best_fitness

    def step(self):
        """Vectorized PSO step."""
        self.ensure_population_evaluated()
        
        if self.velocities is None:
            self.initialize()
            return
        
        # Get current positions as matrix
        X = np.array([np.asarray(sol.representation, dtype=float) for sol in self.population])
        
        # Vectorized velocity update
        r1 = np.random.rand(self.population_size, self._dimension)
        r2 = np.random.rand(self.population_size, self._dimension)
        
        cognitive = self.c1 * r1 * (self.personal_bests - X)
        social = self.c2 * r2 * (self.global_best - X) if self.global_best is not None else 0
        
        self.velocities = self.omega * self.velocities + cognitive + social
        self.velocities = np.clip(self.velocities, -self.vmax, self.vmax)
        
        # Sigmoid and binary update - use steeper sigmoid for more deterministic behavior
        prob = 1.0 / (1.0 + np.exp(-2.0 * self.velocities))  # 2x steeper
        
        # More deterministic update - flip based on velocity direction
        # If velocity strongly points toward 1, set to 1; toward 0, set to 0
        new_X = X.copy().astype(int)
        flip_to_1 = prob > 0.7  # Strong signal for 1
        flip_to_0 = prob < 0.3  # Strong signal for 0
        uncertain = ~flip_to_1 & ~flip_to_0  # In between - use probability
        
        new_X[flip_to_1] = 1
        new_X[flip_to_0] = 0
        new_X[uncertain] = (np.random.rand(*uncertain.shape) < prob)[uncertain]
        
        # Create new solutions and evaluate
        new_population = []
        for i in range(self.population_size):
            sol = Solution(new_X[i], self.problem)
            new_population.append(sol)
        
        self.population = new_population
        self.ensure_population_evaluated()
        
        # Update personal bests
        for i in range(self.population_size):
            if self.population[i].fitness < self.personal_best_fitness[i]:
                self.personal_bests[i] = new_X[i].copy()
                self.personal_best_fitness[i] = self.population[i].fitness
        
        self._update_global_best()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1

    def ingest_population(self, seeds: List[Solution]):
        """Ingest population and re-initialize PSO state."""
        super().ingest_population(seeds)
        
        if not self.population:
            return
        
        self._dimension = len(self.population[0].representation)
        
        # Re-initialize with small velocities
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self._dimension))
        
        self.personal_bests = np.array([np.asarray(sol.representation) for sol in self.population])
        self.personal_best_fitness = np.array([sol.fitness for sol in self.population])
        
        self.global_best_fitness = float('inf')
        self.global_best = None
        self._update_global_best()


# Primary solvers - MAP-Elites for exploration, PSO for exploitation
# These are the recommended choices for the RL orchestrator

# Aliases for backward compatibility (map to new solvers)
GrayWolfOptimization = MAPElitesExplorer  # Use MAP-Elites for exploration
GeneticAlgorithm = BinaryPSO  # Use Binary PSO for exploitation

