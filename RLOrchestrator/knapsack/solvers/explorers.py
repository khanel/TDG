"""
Knapsack Explorer Variants - Configured for DIVERSITY and GLOBAL SEARCH.

All explorers are tuned to:
- Maintain population diversity across the search space
- Favor global exploration over local refinement
- Avoid premature convergence
- Support binary knapsack representation with repair

Key tuning principles:
- High randomness / perturbation
- Low selection pressure
- High mutation rates
- Random injection for diversity
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


# =============================================================================
# Binary Utilities
# =============================================================================

class BinaryKnapsackMixin:
    """Provides binary operations for Knapsack solvers with repair."""
    
    def _repair(self, mask: np.ndarray) -> np.ndarray:
        """Repair mask to satisfy capacity constraint."""
        if hasattr(self.problem, 'repair_mask'):
            return np.asarray(self.problem.repair_mask(mask.tolist()), dtype=int)
        return mask
    
    def _bit_flip_mutation(self, mask: np.ndarray, rate: float) -> np.ndarray:
        """Flip bits with given probability."""
        flips = self.rng.random(len(mask)) < rate
        result = mask.copy()
        result[flips] = 1 - result[flips]
        return self._repair(result)
    
    def _uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover between two binary vectors."""
        mask = self.rng.random(len(parent1)) < 0.5
        child = np.where(mask, parent1, parent2)
        return self._repair(child)
    
    def _two_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Two-point crossover."""
        n = len(parent1)
        p1, p2 = sorted(self.rng.choice(n, 2, replace=False))
        child = parent1.copy()
        child[p1:p2] = parent2[p1:p2]
        return self._repair(child)
    
    def _random_binary(self, dim: int) -> np.ndarray:
        """Generate random binary vector."""
        mask = self.rng.integers(0, 2, size=dim, dtype=int)
        return self._repair(mask)


# =============================================================================
# MAP-Elites Quality-Diversity Explorer
# =============================================================================

@dataclass
class KnapsackElite:
    """Represents an elite solution in the archive."""
    mask: np.ndarray
    fitness: float
    bd: np.ndarray


class KnapsackMapElitesExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    MAP-Elites Quality-Diversity algorithm for Knapsack EXPLORATION.
    
    Maintains archive of diverse, high-quality solutions across
    behavioral space defined by item selection patterns.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 n_bins: int = 10, mutation_rate: float = 0.15,
                 batch_size: int = 32, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.n_bins = n_bins
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self._dim = None
        self.rng = np.random.default_rng(seed)
        self.archive: Dict[Tuple[int, int], KnapsackElite] = {}

    def _compute_behavior_descriptor(self, mask: np.ndarray) -> np.ndarray:
        """Compute 2D behavior descriptor."""
        n = len(mask)
        mid = n // 2
        bd1 = np.mean(mask[:mid])
        bd2 = np.mean(mask[mid:])
        return np.array([bd1, bd2])

    def _bd_to_key(self, bd: np.ndarray) -> Tuple[int, int]:
        bin_x = int(np.clip(bd[0] * self.n_bins, 0, self.n_bins - 1))
        bin_y = int(np.clip(bd[1] * self.n_bins, 0, self.n_bins - 1))
        return (bin_x, bin_y)

    def _add_to_archive(self, mask: np.ndarray, fitness: float) -> bool:
        bd = self._compute_behavior_descriptor(mask)
        key = self._bd_to_key(bd)
        current = self.archive.get(key)
        if current is None or fitness < current.fitness:
            self.archive[key] = KnapsackElite(mask.copy(), fitness, bd)
            return True
        return False

    def initialize(self):
        super().initialize()
        if not self.population:
            return
        self._dim = len(self.population[0].representation)
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()
            self._add_to_archive(np.asarray(sol.representation), sol.fitness)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        if not self.archive:
            return

        elites = list(self.archive.values())
        
        for _ in range(self.batch_size):
            parent = self.rng.choice(elites)
            child = self._bit_flip_mutation(parent.mask, self.mutation_rate)
            sol = Solution(child.tolist(), self.problem)
            sol.evaluate()
            self._add_to_archive(child, sol.fitness)

        # Update population from archive
        self.population = []
        for elite in list(self.archive.values())[:self.population_size]:
            sol = Solution(elite.mask.tolist(), self.problem)
            sol.fitness = elite.fitness
            self.population.append(sol)

        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Genetic Algorithm Explorer
# =============================================================================

class KnapsackGAExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Genetic Algorithm for Knapsack EXPLORATION.
    
    High mutation, low selection pressure for diversity.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 mutation_rate: float = 0.2, crossover_rate: float = 0.8,
                 tournament_size: int = 3, random_injection: float = 0.1,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.random_injection = random_injection
        self._dim = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        new_population = []
        
        # Light elitism
        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        new_population.append(sorted_pop[0].copy(preserve_id=False))

        while len(new_population) < self.population_size:
            if self.rng.random() < self.random_injection:
                mask = self._random_binary(self._dim)
            else:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                if self.rng.random() < self.crossover_rate:
                    mask = self._uniform_crossover(
                        np.array(parent1.representation),
                        np.array(parent2.representation)
                    )
                else:
                    mask = np.array(parent1.representation)
                
                # High mutation for exploration
                mask = self._bit_flip_mutation(mask, self.mutation_rate)

            sol = Solution(mask.tolist(), self.problem)
            sol.evaluate()
            new_population.append(sol)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1

    def _tournament_select(self) -> Solution:
        candidates = self.rng.choice(self.population, self.tournament_size, replace=False)
        return min(candidates, key=lambda s: s.fitness)


# =============================================================================
# PSO Explorer
# =============================================================================

class KnapsackPSOExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Particle Swarm Optimization for Knapsack EXPLORATION.
    
    Binary PSO with high cognitive, low social for exploration.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 cognitive: float = 0.7, social: float = 0.3,
                 mutation_rate: float = 0.15, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.cognitive = cognitive
        self.social = social
        self.mutation_rate = mutation_rate
        self._dim = None
        self.rng = np.random.default_rng(seed)
        self.pbest: List[Solution] = []
        self.gbest: Optional[Solution] = None

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
            for sol in self.population:
                if sol.fitness is None:
                    sol.evaluate()
            self.pbest = [sol.copy(preserve_id=False) for sol in self.population]
            self.gbest = min(self.pbest, key=lambda s: s.fitness)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        for i, sol in enumerate(self.population):
            mask = np.array(sol.representation)
            
            # Move toward personal best
            if self.rng.random() < self.cognitive:
                mask = self._uniform_crossover(mask, np.array(self.pbest[i].representation))
            
            # Move toward global best
            if self.rng.random() < self.social and self.gbest:
                mask = self._uniform_crossover(mask, np.array(self.gbest.representation))
            
            # Random mutation for exploration
            mask = self._bit_flip_mutation(mask, self.mutation_rate)

            new_sol = Solution(mask.tolist(), self.problem)
            new_sol.evaluate()
            self.population[i] = new_sol

            if new_sol.fitness < self.pbest[i].fitness:
                self.pbest[i] = new_sol.copy(preserve_id=False)

        current_best = min(self.population, key=lambda s: s.fitness)
        if self.gbest is None or current_best.fitness < self.gbest.fitness:
            self.gbest = current_best.copy(preserve_id=False)

        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# GWO Explorer
# =============================================================================

class KnapsackGWOExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Grey Wolf Optimizer for Knapsack EXPLORATION.
    
    Weak hierarchy, high randomness for exploration.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 a_decay: float = 0.5, random_weight: float = 0.4,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.a_decay = a_decay
        self.random_weight = random_weight
        self._dim = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        alpha = sorted_pop[0] if len(sorted_pop) > 0 else None
        beta = sorted_pop[1] if len(sorted_pop) > 1 else alpha
        delta = sorted_pop[2] if len(sorted_pop) > 2 else beta

        new_population = []
        for sol in self.population:
            mask = np.array(sol.representation)
            
            if self.rng.random() < self.random_weight:
                mask = self._random_binary(self._dim)
            else:
                r = self.rng.random()
                if r < 0.33 and alpha:
                    mask = self._uniform_crossover(mask, np.array(alpha.representation))
                elif r < 0.66 and beta:
                    mask = self._uniform_crossover(mask, np.array(beta.representation))
                elif delta:
                    mask = self._uniform_crossover(mask, np.array(delta.representation))
                
                mask = self._bit_flip_mutation(mask, 0.15)

            new_sol = Solution(mask.tolist(), self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# ABC Explorer
# =============================================================================

class KnapsackABCExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Artificial Bee Colony for Knapsack EXPLORATION.
    
    Aggressive scouting for diversity.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 limit: int = 5, scout_rate: float = 0.2,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.limit = limit
        self.scout_rate = scout_rate
        self._dim = None
        self.rng = np.random.default_rng(seed)
        self.trials: List[int] = []

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
            self.trials = [0] * len(self.population)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        # Employed bee phase
        for i, sol in enumerate(self.population):
            mask = np.array(sol.representation)
            new_mask = self._bit_flip_mutation(mask, 0.15)
            
            new_sol = Solution(new_mask.tolist(), self.problem)
            new_sol.evaluate()
            
            if new_sol.fitness < sol.fitness:
                self.population[i] = new_sol
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        # Scout phase (aggressive)
        for i in range(len(self.population)):
            if self.trials[i] > self.limit or self.rng.random() < self.scout_rate:
                mask = self._random_binary(self._dim)
                self.population[i] = Solution(mask.tolist(), self.problem)
                self.population[i].evaluate()
                self.trials[i] = 0

        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# WOA Explorer
# =============================================================================

class KnapsackWOAExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Whale Optimization Algorithm for Knapsack EXPLORATION.
    
    Extended random search for diversity.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 random_search_prob: float = 0.4, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.random_search_prob = random_search_prob
        self._dim = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        best = min(self.population, key=lambda s: s.fitness)

        new_population = []
        for sol in self.population:
            mask = np.array(sol.representation)
            
            if self.rng.random() < self.random_search_prob:
                random_sol = self.rng.choice(self.population)
                mask = self._uniform_crossover(mask, np.array(random_sol.representation))
            else:
                mask = self._uniform_crossover(mask, np.array(best.representation))
            
            mask = self._bit_flip_mutation(mask, 0.15)
            
            new_sol = Solution(mask.tolist(), self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# HHO Explorer
# =============================================================================

class KnapsackHHOExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Harris Hawks Optimization for Knapsack EXPLORATION.
    
    Soft besiege emphasis for diversity.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 energy_decay: float = 0.5, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.energy_decay = energy_decay
        self._dim = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        E = 2 * self.rng.random() - 1
        rabbit = min(self.population, key=lambda s: s.fitness)

        new_population = []
        for sol in self.population:
            mask = np.array(sol.representation)
            
            if abs(E) >= 1:
                # Exploration phase
                if self.rng.random() < 0.5:
                    mask = self._random_binary(self._dim)
                else:
                    random_sol = self.rng.choice(self.population)
                    mask = self._uniform_crossover(mask, np.array(random_sol.representation))
            else:
                # Soft besiege
                mask = self._uniform_crossover(mask, np.array(rabbit.representation))
                mask = self._bit_flip_mutation(mask, 0.2)
            
            new_sol = Solution(mask.tolist(), self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# MPA Explorer
# =============================================================================

class KnapsackMPAExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Marine Predators Algorithm for Knapsack EXPLORATION.
    
    Lévy flights dominant for exploration.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 FADs: float = 0.3, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.FADs = FADs
        self._dim = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        elite = min(self.population, key=lambda s: s.fitness)

        new_population = []
        for sol in self.population:
            mask = np.array(sol.representation)
            
            # Lévy flight (big random jumps)
            if self.rng.random() < 0.5:
                n_flips = self.rng.integers(3, max(4, self._dim // 5))
                indices = self.rng.choice(self._dim, n_flips, replace=False)
                mask[indices] = 1 - mask[indices]
                mask = self._repair(mask)
            else:
                mask = self._uniform_crossover(mask, np.array(elite.representation))
                mask = self._bit_flip_mutation(mask, 0.15)
            
            # FADs effect
            if self.rng.random() < self.FADs:
                mask = self._bit_flip_mutation(mask, 0.1)

            new_sol = Solution(mask.tolist(), self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# SMA Explorer
# =============================================================================

class KnapsackSMAExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Slime Mould Algorithm for Knapsack EXPLORATION.
    
    Weak feedback, high randomness.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 z: float = 0.03, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.z = z
        self._dim = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        best = min(self.population, key=lambda s: s.fitness)

        new_population = []
        for sol in self.population:
            mask = np.array(sol.representation)
            
            if self.rng.random() < self.z:
                mask = self._random_binary(self._dim)
            else:
                if self.rng.random() < 0.5:
                    mask = self._uniform_crossover(mask, np.array(best.representation))
                else:
                    rand_sol = self.rng.choice(self.population)
                    mask = self._uniform_crossover(mask, np.array(rand_sol.representation))
                
                mask = self._bit_flip_mutation(mask, 0.15)

            new_sol = Solution(mask.tolist(), self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# GSA Explorer
# =============================================================================

class KnapsackGSAExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Gravitational Search Algorithm for Knapsack EXPLORATION.
    
    Low gravitational constant for weak attraction.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 G0: float = 50.0, alpha: float = 10.0,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.G0 = G0
        self.alpha = alpha
        self._dim = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        G = self.G0 * np.exp(-self.alpha * self.iteration / 100)
        
        fitnesses = np.array([sol.fitness for sol in self.population])
        worst = fitnesses.max()
        best_fit = fitnesses.min()
        
        if worst - best_fit > 1e-10:
            masses = (worst - fitnesses) / (worst - best_fit)
        else:
            masses = np.ones(len(self.population))
        masses = masses / masses.sum()

        new_population = []
        for i, sol in enumerate(self.population):
            mask = np.array(sol.representation)
            
            for j, other in enumerate(self.population):
                if i != j and masses[j] > 0.1:
                    if self.rng.random() < G * masses[j] * 0.3:
                        mask = self._uniform_crossover(mask, np.array(other.representation))
                        break
            
            mask = self._bit_flip_mutation(mask, 0.15)

            new_sol = Solution(mask.tolist(), self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Diversity Explorer (Custom)
# =============================================================================

class KnapsackDiversityExplorer(SearchAlgorithm, BinaryKnapsackMixin):
    """
    Custom diversity-maintaining explorer for Knapsack.
    
    Explicitly maintains population diversity using Hamming distance.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 sigma: float = 0.3, alpha: float = 1.0,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.sigma = sigma
        self.alpha = alpha
        self._dim = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._dim = len(self.population[0].representation)
        self._update_best_solution()

    def _hamming_distance(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute normalized Hamming distance."""
        return np.mean(mask1 != mask2)

    def step(self):
        for sol in self.population:
            if sol.fitness is None:
                sol.evaluate()

        masks = [np.array(sol.representation) for sol in self.population]
        shared_fitness = []
        
        for i, sol in enumerate(self.population):
            niche_count = 0.0
            for j in range(len(self.population)):
                d = self._hamming_distance(masks[i], masks[j])
                if d < self.sigma:
                    niche_count += 1.0 - (d / self.sigma) ** self.alpha
            shared_fitness.append(sol.fitness * max(1.0, niche_count))

        new_population = []
        
        # Keep best
        best_idx = np.argmin([sol.fitness for sol in self.population])
        new_population.append(self.population[best_idx].copy(preserve_id=False))

        while len(new_population) < self.population_size:
            candidates = self.rng.choice(len(self.population), 3, replace=False)
            winner = min(candidates, key=lambda x: shared_fitness[x])
            
            mask = np.array(self.population[winner].representation)
            
            if self.rng.random() < 0.7:
                other = self.rng.choice(self.population)
                mask = self._uniform_crossover(mask, np.array(other.representation))
            
            mask = self._bit_flip_mutation(mask, 0.2)

            new_sol = Solution(mask.tolist(), self.problem)
            new_sol.evaluate()
            new_population.append(new_sol)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1
