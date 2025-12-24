"""
TSP Explorer Variants - Configured for DIVERSITY and GLOBAL SEARCH.

All explorers are tuned to:
- Maintain tour diversity across the search space
- Favor global exploration over local refinement
- Avoid premature convergence
- Support permutation-based TSP representation

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
# Permutation Utilities
# =============================================================================

class PermutationMixin:
    """Provides permutation operations for TSP solvers."""

    def _repair_permutation(self, tour: np.ndarray) -> np.ndarray:
        """Repair an arbitrary vector into a valid permutation of 0..n-1.

        Some solver initializations may produce duplicates or out-of-range values.
        MAP-Elites behavior descriptors and archive indexing assume a proper
        permutation tour.
        """
        arr = np.asarray(tour)
        n = int(arr.shape[0])
        if n <= 0:
            return arr.astype(int)

        # Fast path: already a permutation.
        if arr.dtype.kind in {"i", "u"}:
            vals = arr.astype(int, copy=False)
            if vals.min(initial=0) >= 0 and vals.max(initial=-1) < n and len(set(vals.tolist())) == n:
                return vals

        vals = np.asarray(np.rint(arr), dtype=int)
        used: set[int] = set()
        repaired = np.full(n, -1, dtype=int)
        for idx, v in enumerate(vals.tolist()):
            if 0 <= v < n and v not in used:
                repaired[idx] = v
                used.add(v)

        missing = [i for i in range(n) if i not in used]
        miss_idx = 0
        for idx in range(n):
            if repaired[idx] == -1:
                repaired[idx] = missing[miss_idx]
                miss_idx += 1
        return repaired
    
    def _order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Order Crossover (OX) for permutations."""
        n = len(parent1)
        start, end = sorted(np.random.choice(n, 2, replace=False))
        child = np.full(n, -1)
        child[start:end+1] = parent1[start:end+1]
        segment_set = set(child[start:end+1])
        p2_remaining = [x for x in parent2 if x not in segment_set]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = p2_remaining[idx]
                idx += 1
        return child

    def _pmx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Partially Mapped Crossover (PMX)."""
        n = len(parent1)
        start, end = sorted(np.random.choice(n, 2, replace=False))
        child = np.full(n, -1)
        child[start:end+1] = parent1[start:end+1]
        mapping = {parent1[i]: parent2[i] for i in range(start, end+1)}
        
        for i in range(n):
            if child[i] == -1:
                val = parent2[i]
                while val in child[start:end+1]:
                    val = mapping.get(val, val)
                child[i] = val
        return child

    def _two_opt_move(self, tour: np.ndarray) -> np.ndarray:
        """Apply a single 2-opt move (reverse a segment)."""
        n = len(tour)
        i, j = sorted(np.random.choice(n, 2, replace=False))
        new_tour = tour.copy()
        new_tour[i:j+1] = new_tour[i:j+1][::-1]
        return new_tour

    def _or_opt_move(self, tour: np.ndarray) -> np.ndarray:
        """Or-opt: relocate a segment of 1-3 cities."""
        n = len(tour)
        seg_len = np.random.randint(1, min(4, n-1))
        start = np.random.randint(0, n - seg_len)
        segment = tour[start:start+seg_len].copy()
        remaining = np.concatenate([tour[:start], tour[start+seg_len:]])
        insert_pos = np.random.randint(0, len(remaining) + 1)
        return np.concatenate([remaining[:insert_pos], segment, remaining[insert_pos:]])

    def _swap_move(self, tour: np.ndarray) -> np.ndarray:
        """Swap two random cities."""
        n = len(tour)
        i, j = np.random.choice(n, 2, replace=False)
        new_tour = tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def _scramble_mutation(self, tour: np.ndarray, rate: float = 0.2) -> np.ndarray:
        """Scramble a random segment."""
        n = len(tour)
        seg_len = max(2, int(n * rate))
        start = np.random.randint(0, n - seg_len + 1)
        new_tour = tour.copy()
        segment = new_tour[start:start+seg_len].copy()
        np.random.shuffle(segment)
        new_tour[start:start+seg_len] = segment
        return new_tour


# =============================================================================
# MAP-Elites Quality-Diversity Explorer
# =============================================================================

@dataclass
class TSPElite:
    """Represents an elite tour in the archive."""
    tour: np.ndarray
    fitness: float
    bd: np.ndarray  # Behavior descriptor


class TSPMapElitesExplorer(SearchAlgorithm, PermutationMixin):
    """
    MAP-Elites Quality-Diversity algorithm for TSP EXPLORATION.
    
    Maintains archive of diverse, high-quality tours across
    a behavioral space defined by tour characteristics.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int, 
                 n_bins: int = 10, mutation_rate: float = 0.3,
                 batch_size: int = 32, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.n_bins = n_bins
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self._num_cities = None
        self.rng = np.random.default_rng(seed)
        self.archive: Dict[Tuple[int, int], TSPElite] = {}

    def _compute_behavior_descriptor(self, tour: np.ndarray) -> np.ndarray:
        """Compute 2D behavior descriptor for tour."""
        tour = self._repair_permutation(tour)
        n = len(tour)
        mid = n // 2
        # BD1: average position of first half cities in tour
        bd1 = np.mean([np.where(tour == i)[0][0] / n for i in range(mid)])
        # BD2: how "clustered" the tour is (variance of positions)
        positions = np.arange(n) / n
        bd2 = np.std(positions[tour[:mid]]) if mid > 0 else 0.5
        return np.array([bd1, bd2])

    def _bd_to_key(self, bd: np.ndarray) -> Tuple[int, int]:
        """Convert behavior descriptor to archive key."""
        bin_x = int(np.clip(bd[0] * self.n_bins, 0, self.n_bins - 1))
        bin_y = int(np.clip(bd[1] * self.n_bins, 0, self.n_bins - 1))
        return (bin_x, bin_y)

    def _add_to_archive(self, tour: np.ndarray, fitness: float) -> bool:
        """Add tour to archive if it improves the cell."""
        tour = self._repair_permutation(tour)
        bd = self._compute_behavior_descriptor(tour)
        key = self._bd_to_key(bd)
        current = self.archive.get(key)
        if current is None or fitness < current.fitness:
            self.archive[key] = TSPElite(tour.copy(), fitness, bd)
            return True
        return False

    def initialize(self):
        """Initialize population and archive."""
        super().initialize()
        if not self.population:
            return
        self._num_cities = len(self.population[0].representation)
        for sol in self.population:
            self._add_to_archive(np.asarray(sol.representation), sol.fitness)
        self._update_best_solution()

    def step(self):
        """One step of MAP-Elites."""
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        # Generate offspring from archive
        if not self.archive:
            return

        elites = list(self.archive.values())
        new_tours = []
        
        for _ in range(self.batch_size):
            parent = self.rng.choice(elites)
            child = parent.tour.copy()
            
            # Apply random mutations
            if self.rng.random() < self.mutation_rate:
                child = self._two_opt_move(child)
            if self.rng.random() < self.mutation_rate * 0.5:
                child = self._or_opt_move(child)
            if self.rng.random() < self.mutation_rate * 0.3:
                child = self._scramble_mutation(child, 0.15)
            
            new_tours.append(child)

        # Evaluate and add to archive
        for tour in new_tours:
            tour = self._repair_permutation(tour)
            sol = Solution(tour.tolist(), self.problem)
            sol.evaluate()
            self._add_to_archive(tour, sol.fitness)

        # Update population from archive
        self.population = []
        for elite in list(self.archive.values())[:self.population_size]:
            sol = Solution(elite.tour.tolist(), self.problem)
            sol.fitness = elite.fitness
            self.population.append(sol)

        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Genetic Algorithm Explorer
# =============================================================================

class TSPGAExplorer(SearchAlgorithm, PermutationMixin):
    """
    Genetic Algorithm for TSP EXPLORATION.
    
    High mutation, low selection pressure for diversity.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 mutation_rate: float = 0.4, crossover_rate: float = 0.8,
                 tournament_size: int = 3, random_injection: float = 0.1,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.random_injection = random_injection
        self._num_cities = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        new_population = []
        
        # Keep best (light elitism)
        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        new_population.append(sorted_pop[0].copy(preserve_id=False))

        while len(new_population) < self.population_size:
            if self.rng.random() < self.random_injection:
                # Random tour injection
                tour = self.rng.permutation(self._num_cities)
            else:
                # Tournament selection
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                if self.rng.random() < self.crossover_rate:
                    tour = self._order_crossover(
                        np.array(parent1.representation),
                        np.array(parent2.representation)
                    )
                else:
                    tour = np.array(parent1.representation)
                
                # Multiple mutations for exploration
                if self.rng.random() < self.mutation_rate:
                    tour = self._two_opt_move(tour)
                if self.rng.random() < self.mutation_rate * 0.5:
                    tour = self._swap_move(tour)
                if self.rng.random() < self.mutation_rate * 0.3:
                    tour = self._scramble_mutation(tour, 0.2)

            sol = Solution(list(tour), self.problem)
            new_population.append(sol)

        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1

    def _tournament_select(self) -> Solution:
        """Weak tournament selection for low pressure."""
        candidates = self.rng.choice(self.population, self.tournament_size, replace=False)
        return min(candidates, key=lambda s: s.fitness)


# =============================================================================
# PSO Explorer
# =============================================================================

class TSPPSOExplorer(SearchAlgorithm, PermutationMixin):
    """
    Particle Swarm Optimization for TSP EXPLORATION.
    
    Low inertia, high cognitive component for exploration.
    Uses swap sequences as velocity representation.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 cognitive: float = 0.7, social: float = 0.3,
                 random_moves: int = 3, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.cognitive = cognitive
        self.social = social
        self.random_moves = random_moves
        self._num_cities = None
        self.rng = np.random.default_rng(seed)
        self.pbest: List[Solution] = []
        self.gbest: Optional[Solution] = None

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
            self.pbest = [sol.copy(preserve_id=False) for sol in self.population]
            self.gbest = min(self.pbest, key=lambda s: s.fitness)
        self._update_best_solution()

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        for i, sol in enumerate(self.population):
            tour = np.array(sol.representation)
            
            # Move toward personal best
            if self.rng.random() < self.cognitive:
                tour = self._partial_crossover(tour, np.array(self.pbest[i].representation))
            
            # Move toward global best
            if self.rng.random() < self.social and self.gbest:
                tour = self._partial_crossover(tour, np.array(self.gbest.representation))
            
            # Random exploration moves
            for _ in range(self.random_moves):
                if self.rng.random() < 0.5:
                    tour = self._two_opt_move(tour)
                else:
                    tour = self._swap_move(tour)

            new_sol = Solution(tour.tolist(), self.problem)
            new_sol.evaluate()
            self.population[i] = new_sol

            # Update personal best
            if new_sol.fitness < self.pbest[i].fitness:
                self.pbest[i] = new_sol.copy(preserve_id=False)

        # Update global best
        current_best = min(self.population, key=lambda s: s.fitness)
        if self.gbest is None or current_best.fitness < self.gbest.fitness:
            self.gbest = current_best.copy(preserve_id=False)

        self._update_best_solution()
        self.iteration += 1

    def _partial_crossover(self, tour1: np.ndarray, tour2: np.ndarray) -> np.ndarray:
        """Apply partial crossover to move tour1 toward tour2."""
        n = len(tour1)
        seg_len = self.rng.integers(2, max(3, n // 3))
        start = self.rng.integers(0, n - seg_len)
        return self._order_crossover(tour1, tour2)


# =============================================================================
# Grey Wolf Optimizer Explorer
# =============================================================================

class TSPGWOExplorer(SearchAlgorithm, PermutationMixin):
    """
    Grey Wolf Optimizer for TSP EXPLORATION.
    
    Weak hierarchy, high randomness for exploration.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 a_decay: float = 0.5, random_weight: float = 0.4,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.a_decay = a_decay
        self.random_weight = random_weight
        self._num_cities = None
        self.rng = np.random.default_rng(seed)
        self.alpha: Optional[Solution] = None
        self.beta: Optional[Solution] = None
        self.delta: Optional[Solution] = None

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
            self._update_hierarchy()
        self._update_best_solution()

    def _update_hierarchy(self):
        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        self.alpha = sorted_pop[0] if len(sorted_pop) > 0 else None
        self.beta = sorted_pop[1] if len(sorted_pop) > 1 else self.alpha
        self.delta = sorted_pop[2] if len(sorted_pop) > 2 else self.beta

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        a = 2.0 - self.iteration * self.a_decay / 100  # Linearly decreased from 2 to 0

        new_population = []
        for sol in self.population:
            tour = np.array(sol.representation)
            
            # High random exploration
            if self.rng.random() < self.random_weight:
                tour = self.rng.permutation(self._num_cities)
            else:
                # Encircle alpha, beta, delta with weak attraction
                r1 = self.rng.random()
                if r1 < 0.33 and self.alpha:
                    tour = self._order_crossover(tour, np.array(self.alpha.representation))
                elif r1 < 0.66 and self.beta:
                    tour = self._order_crossover(tour, np.array(self.beta.representation))
                elif self.delta:
                    tour = self._order_crossover(tour, np.array(self.delta.representation))
                
                # Random perturbation
                tour = self._two_opt_move(tour)
                if self.rng.random() < 0.3:
                    tour = self._scramble_mutation(tour, 0.15)

            new_sol = Solution(tour.tolist(), self.problem)
            new_population.append(new_sol)

        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_hierarchy()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# ABC Explorer
# =============================================================================

class TSPABCExplorer(SearchAlgorithm, PermutationMixin):
    """
    Artificial Bee Colony for TSP EXPLORATION.
    
    Aggressive scouting, high abandonment limit for diversity.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 limit: int = 5, scout_rate: float = 0.2,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.limit = limit
        self.scout_rate = scout_rate
        self._num_cities = None
        self.rng = np.random.default_rng(seed)
        self.trials: List[int] = []

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
            self.trials = [0] * len(self.population)
        self._update_best_solution()

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        # Employed bee phase
        for i, sol in enumerate(self.population):
            tour = np.array(sol.representation)
            new_tour = self._two_opt_move(tour)
            if self.rng.random() < 0.5:
                new_tour = self._or_opt_move(new_tour)
            
            new_sol = Solution(new_tour.tolist(), self.problem)
            new_sol.evaluate()
            
            if new_sol.fitness < sol.fitness:
                self.population[i] = new_sol
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        # Onlooker bee phase (weak selection)
        fitnesses = np.array([1.0 / (1.0 + sol.fitness) for sol in self.population])
        probs = fitnesses / fitnesses.sum()
        
        for _ in range(self.population_size):
            idx = self.rng.choice(len(self.population), p=probs)
            tour = np.array(self.population[idx].representation)
            new_tour = self._two_opt_move(tour)
            
            new_sol = Solution(new_tour.tolist(), self.problem)
            new_sol.evaluate()
            
            if new_sol.fitness < self.population[idx].fitness:
                self.population[idx] = new_sol
                self.trials[idx] = 0

        # Scout bee phase (aggressive)
        for i in range(len(self.population)):
            if self.trials[i] > self.limit or self.rng.random() < self.scout_rate:
                tour = self.rng.permutation(self._num_cities)
                self.population[i] = Solution(tour.tolist(), self.problem)
                self.trials[i] = 0

        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# WOA Explorer
# =============================================================================

class TSPWOAExplorer(SearchAlgorithm, PermutationMixin):
    """
    Whale Optimization Algorithm for TSP EXPLORATION.
    
    Extended spiral search, high randomness.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 b: float = 1.0, a_decay: float = 0.5,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.b = b
        self.a_decay = a_decay
        self._num_cities = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        a = 2.0 - self.iteration * self.a_decay / 100
        best = min(self.population, key=lambda s: s.fitness)

        new_population = []
        for sol in self.population:
            tour = np.array(sol.representation)
            p = self.rng.random()
            
            if p < 0.5:
                # Extended random search
                if self.rng.random() < 0.4:
                    tour = self.rng.permutation(self._num_cities)
                else:
                    random_sol = self.rng.choice(self.population)
                    tour = self._order_crossover(tour, np.array(random_sol.representation))
            else:
                # Spiral update toward best
                tour = self._order_crossover(tour, np.array(best.representation))
            
            # Always apply perturbation
            tour = self._two_opt_move(tour)
            if self.rng.random() < 0.3:
                tour = self._scramble_mutation(tour, 0.2)

            new_sol = Solution(tour.tolist(), self.problem)
            new_population.append(new_sol)

        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# HHO Explorer
# =============================================================================

class TSPHHOExplorer(SearchAlgorithm, PermutationMixin):
    """
    Harris Hawks Optimization for TSP EXPLORATION.
    
    Soft besiege emphasis for diversity.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 energy_decay: float = 0.5, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.energy_decay = energy_decay
        self._num_cities = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        E0 = 2 * self.rng.random() - 1
        E = E0 * (1 - self.iteration / 100 * self.energy_decay)
        rabbit = min(self.population, key=lambda s: s.fitness)

        new_population = []
        for sol in self.population:
            tour = np.array(sol.representation)
            
            if abs(E) >= 1:
                # Exploration: random search
                if self.rng.random() < 0.5:
                    tour = self.rng.permutation(self._num_cities)
                else:
                    random_sol = self.rng.choice(self.population)
                    tour = self._order_crossover(tour, np.array(random_sol.representation))
            else:
                # Soft besiege (exploration-biased)
                tour = self._order_crossover(tour, np.array(rabbit.representation))
                tour = self._scramble_mutation(tour, 0.25)
            
            tour = self._two_opt_move(tour)
            new_sol = Solution(tour.tolist(), self.problem)
            new_population.append(new_sol)

        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# MPA Explorer
# =============================================================================

class TSPMPAExplorer(SearchAlgorithm, PermutationMixin):
    """
    Marine Predators Algorithm for TSP EXPLORATION.
    
    Lévy flights dominant for exploration.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 FADs: float = 0.3, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.FADs = FADs
        self._num_cities = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        new_population = []
        elite = min(self.population, key=lambda s: s.fitness)

        for sol in self.population:
            tour = np.array(sol.representation)
            
            # Lévy flight exploration
            if self.rng.random() < 0.5:
                # Big random jumps
                n_swaps = self.rng.integers(3, max(4, self._num_cities // 5))
                for _ in range(n_swaps):
                    tour = self._swap_move(tour)
            else:
                # Move toward elite with perturbation
                tour = self._order_crossover(tour, np.array(elite.representation))
                tour = self._scramble_mutation(tour, 0.2)
            
            # FADs effect
            if self.rng.random() < self.FADs:
                tour = self._or_opt_move(tour)

            new_sol = Solution(tour.tolist(), self.problem)
            new_population.append(new_sol)

        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# SMA Explorer
# =============================================================================

class TSPSMAExplorer(SearchAlgorithm, PermutationMixin):
    """
    Slime Mould Algorithm for TSP EXPLORATION.
    
    Weak feedback, high randomness.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 z: float = 0.03, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.z = z
        self._num_cities = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        best = sorted_pop[0]

        new_population = []
        for i, sol in enumerate(self.population):
            tour = np.array(sol.representation)
            
            if self.rng.random() < self.z:
                # Random search
                tour = self.rng.permutation(self._num_cities)
            else:
                # Weak attraction to best/random
                if self.rng.random() < 0.5:
                    tour = self._order_crossover(tour, np.array(best.representation))
                else:
                    rand_sol = self.rng.choice(self.population)
                    tour = self._order_crossover(tour, np.array(rand_sol.representation))
                
                tour = self._two_opt_move(tour)
                if self.rng.random() < 0.3:
                    tour = self._scramble_mutation(tour, 0.15)

            new_sol = Solution(tour.tolist(), self.problem)
            new_population.append(new_sol)

        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# GSA Explorer
# =============================================================================

class TSPGSAExplorer(SearchAlgorithm, PermutationMixin):
    """
    Gravitational Search Algorithm for TSP EXPLORATION.
    
    Low gravitational constant for weak attraction.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 G0: float = 50.0, alpha: float = 10.0,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.G0 = G0
        self.alpha = alpha
        self._num_cities = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        G = self.G0 * np.exp(-self.alpha * self.iteration / 100)
        
        # Compute masses (inverted fitness)
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
            tour = np.array(sol.representation)
            
            # Move toward heavy masses (weak attraction)
            for j, other in enumerate(self.population):
                if i != j and masses[j] > 0.1:
                    if self.rng.random() < G * masses[j] * 0.3:
                        tour = self._order_crossover(tour, np.array(other.representation))
                        break
            
            # Random perturbation
            tour = self._two_opt_move(tour)
            if self.rng.random() < 0.4:
                tour = self._swap_move(tour)

            new_sol = Solution(tour.tolist(), self.problem)
            new_population.append(new_sol)

        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Diversity Explorer (Custom)
# =============================================================================

class TSPDiversityExplorer(SearchAlgorithm, PermutationMixin):
    """
    Custom diversity-maintaining explorer for TSP.
    
    Explicitly maintains population diversity using fitness sharing.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int,
                 sigma: float = 0.3, alpha: float = 1.0,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.sigma = sigma
        self.alpha = alpha
        self._num_cities = None
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        if self.population:
            self._num_cities = len(self.population[0].representation)
        self._update_best_solution()

    def _tour_distance(self, tour1: np.ndarray, tour2: np.ndarray) -> float:
        """Compute normalized distance between tours (edge difference)."""
        edges1 = set(zip(tour1, np.roll(tour1, -1)))
        edges2 = set(zip(tour2, np.roll(tour2, -1)))
        common = len(edges1 & edges2)
        return 1.0 - common / len(edges1)

    def step(self):
        self.ensure_population_evaluated()
        
        if self._num_cities is None and self.population:
            self._num_cities = len(self.population[0].representation)

        # Compute shared fitness
        tours = [np.array(sol.representation) for sol in self.population]
        shared_fitness = []
        
        for i, sol in enumerate(self.population):
            niche_count = 0.0
            for j, other in enumerate(self.population):
                d = self._tour_distance(tours[i], tours[j])
                if d < self.sigma:
                    niche_count += 1.0 - (d / self.sigma) ** self.alpha
            shared_fitness.append(sol.fitness * max(1.0, niche_count))

        # Selection based on shared fitness
        new_population = []
        
        # Keep most diverse + best
        best_idx = np.argmin([sol.fitness for sol in self.population])
        new_population.append(self.population[best_idx].copy(preserve_id=False))

        while len(new_population) < self.population_size:
            # Tournament on shared fitness
            candidates = self.rng.choice(len(self.population), 3, replace=False)
            winner = min(candidates, key=lambda x: shared_fitness[x])
            
            tour = np.array(self.population[winner].representation)
            
            # Crossover with random
            if self.rng.random() < 0.7:
                other = self.rng.choice(self.population)
                tour = self._order_crossover(tour, np.array(other.representation))
            
            # Mutations
            tour = self._two_opt_move(tour)
            if self.rng.random() < 0.3:
                tour = self._scramble_mutation(tour, 0.2)

            new_sol = Solution(tour.tolist(), self.problem)
            new_population.append(new_sol)

        self.population = new_population
        self.mark_best_dirty()
        self.ensure_population_evaluated()
        self._update_best_solution()
        self.iteration += 1
