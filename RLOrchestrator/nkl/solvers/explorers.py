"""
NKL Explorer Variants - Configured for DIVERSITY and GLOBAL SEARCH.

All explorers are tuned to:
- Maintain population diversity
- Favor global search over local refinement
- Avoid premature convergence
- Support binary NKL representation

Key tuning principles:
- High randomness / perturbation
- Low selection pressure
- High mutation rates
- Weak attraction to best solutions
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


# =============================================================================
# Binary Conversion Mixin
# =============================================================================

class BinaryMixin:
    """Provides binary conversion for continuous-valued algorithms."""
    
    def _to_binary(self, vector: np.ndarray) -> np.ndarray:
        """Convert continuous [0,1] vector to binary using threshold."""
        return (np.asarray(vector) >= 0.5).astype(int)
    
    def _to_continuous(self, binary: np.ndarray) -> np.ndarray:
        """Convert binary to continuous (just cast to float)."""
        return np.asarray(binary, dtype=float)
    
    def _binary_solution(self, vector: np.ndarray) -> Solution:
        """Create binary solution from continuous vector."""
        binary = self._to_binary(vector)
        sol = Solution(binary.tolist(), self.problem)
        sol.evaluate()
        return sol
    
    def _random_binary(self, dim: int) -> np.ndarray:
        """Generate random binary vector."""
        return self.rng.integers(0, 2, size=dim, dtype=int)
    
    def _bit_flip_mutation(self, binary: np.ndarray, rate: float) -> np.ndarray:
        """Flip bits with given probability."""
        mask = self.rng.random(len(binary)) < rate
        result = binary.copy()
        result[mask] = 1 - result[mask]
        return result


# =============================================================================
# MAP-Elites Quality-Diversity Explorer
# =============================================================================

@dataclass
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
                 batch_size: int = 32, seed: Optional[int] = None, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.n_bins = n_bins
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self._dimension = None
        self.rng = np.random.default_rng(seed)
        
        # Archive: maps (bin_x, bin_y) -> Elite
        self.archive: Dict[Tuple[int, int], Elite] = {}
        
    def _compute_behavior_descriptor(self, x: np.ndarray) -> np.ndarray:
        """Compute 2D behavior descriptor for binary solution."""
        mid = len(x) // 2
        bd1 = np.mean(x[:mid])
        bd2 = np.mean(x[mid:])
        return np.array([bd1, bd2])
    
    def _bd_to_key(self, bd: np.ndarray) -> Tuple[int, int]:
        """Convert behavior descriptor to archive key."""
        bin_x = int(np.clip(bd[0] * self.n_bins, 0, self.n_bins - 1))
        bin_y = int(np.clip(bd[1] * self.n_bins, 0, self.n_bins - 1))
        return (bin_x, bin_y)
    
    def _add_to_archive(self, x: np.ndarray, fitness: float) -> bool:
        """Add solution to archive if it improves the cell."""
        bd = self._compute_behavior_descriptor(x)
        key = self._bd_to_key(bd)
        
        current = self.archive.get(key)
        if current is None or fitness < current.fitness:
            self.archive[key] = Elite(x.copy(), fitness, bd)
            return True
        return False

    def initialize(self):
        """Initialize population and archive."""
        super().initialize()
        
        if not self.population:
            return
            
        self._dimension = len(self.population[0].representation)
        
        for sol in self.population:
            self._add_to_archive(np.asarray(sol.representation), sol.fitness)
        
        self._update_best_solution()

    def step(self):
        """One step of MAP-Elites."""
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        if not self.archive:
            self.initialize()
            return
        
        archive_elites = list(self.archive.values())
        
        new_solutions = []
        for _ in range(self.batch_size):
            parent = archive_elites[self.rng.integers(len(archive_elites))]
            parent_x = parent.x.copy()
            
            mutation_mask = self.rng.random(self._dimension) < self.mutation_rate
            child_x = parent_x.copy()
            child_x[mutation_mask] = 1 - child_x[mutation_mask]
            
            child_sol = Solution(child_x.astype(int).tolist(), self.problem)
            child_sol.evaluate()
            
            self._add_to_archive(child_x, child_sol.fitness)
            new_solutions.append(child_sol)
        
        archive_solutions = []
        for elite in self.archive.values():
            sol = Solution(elite.x.astype(int).tolist(), self.problem)
            sol.fitness = elite.fitness
            archive_solutions.append(sol)
        
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


# =============================================================================
# Grey Wolf Optimizer - Explorer Variant
# =============================================================================

class NKLGWOExplorer(BinaryMixin, SearchAlgorithm):
    """
    GWO configured for EXPLORATION.
    
    Exploration tuning:
    - High 'a' decay start (3.0 instead of 2.0) - more randomness
    - Wider position updates - wolves spread out more
    - Random leader selection occasionally
    - High mutation after position update
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        a_initial: float = 3.0,      # Higher = more exploration (standard is 2.0)
        a_final: float = 1.0,        # Don't go to 0, maintain some randomness
        mutation_rate: float = 0.15,  # Post-update mutation for diversity
        random_leader_prob: float = 0.3,  # Probability to follow random wolf
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.a_initial = a_initial
        self.a_final = a_final
        self.mutation_rate = mutation_rate
        self.random_leader_prob = random_leader_prob
        self._dimension = None
        self.max_iterations = kwargs.get('max_iterations', 1000)
    
    def initialize(self):
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda s: s.fitness)
        alpha = np.asarray(sorted_pop[0].representation, dtype=float)
        beta = np.asarray(sorted_pop[1].representation, dtype=float)
        delta = np.asarray(sorted_pop[2].representation, dtype=float)
        
        # Decay 'a' but keep it higher than standard GWO
        progress = min(1.0, self.iteration / max(1, self.max_iterations))
        a = self.a_initial - (self.a_initial - self.a_final) * progress
        
        new_population = []
        for wolf in self.population:
            x = np.asarray(wolf.representation, dtype=float)
            
            # Occasionally follow a random wolf instead of leaders (exploration)
            if self.rng.random() < self.random_leader_prob:
                random_wolf = self.population[self.rng.integers(len(self.population))]
                leader = np.asarray(random_wolf.representation, dtype=float)
                r1, r2 = self.rng.random(self._dimension), self.rng.random(self._dimension)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - x)
                new_x = leader - A * D
            else:
                # Standard GWO update with higher 'a'
                X1 = self._update_position(x, alpha, a)
                X2 = self._update_position(x, beta, a)
                X3 = self._update_position(x, delta, a)
                new_x = (X1 + X2 + X3) / 3
            
            # Clip to [0, 1]
            new_x = np.clip(new_x, 0, 1)
            
            # Convert to binary
            binary = self._to_binary(new_x)
            
            # Apply mutation for diversity
            binary = self._bit_flip_mutation(binary, self.mutation_rate)
            
            new_sol = Solution(binary.tolist(), self.problem)
            new_population.append(new_sol)
        
        self.population = new_population
        self.ensure_population_evaluated()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1
    
    def _update_position(self, wolf, leader, a):
        r1 = self.rng.random(self._dimension)
        r2 = self.rng.random(self._dimension)
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * leader - wolf)
        return leader - A * D


# =============================================================================
# Particle Swarm Optimization - Explorer Variant
# =============================================================================

class NKLPSOExplorer(BinaryMixin, SearchAlgorithm):
    """
    PSO configured for EXPLORATION.
    
    Exploration tuning:
    - High inertia weight (0.9) - particles maintain momentum
    - High cognitive coefficient (c1=2.5) - personal exploration
    - Low social coefficient (c2=0.5) - weak swarm attraction
    - High velocity limits - larger jumps
    - Random reinitialization for stagnant particles
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        omega: float = 0.9,          # High inertia for momentum
        c1: float = 2.5,             # High cognitive (personal best)
        c2: float = 0.5,             # Low social (global best)
        vmax: float = 6.0,           # High velocity limit
        reinit_stagnation: int = 10, # Reinit after N steps without improvement
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.vmax = vmax
        self.reinit_stagnation = reinit_stagnation
        
        self.velocities = None
        self.personal_bests = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.stagnation_counters = None
        self._dimension = None
    
    def initialize(self):
        super().initialize()
        if not self.population:
            return
        
        self._dimension = len(self.population[0].representation)
        
        # Initialize velocities with high variance for exploration
        self.velocities = self.rng.uniform(-self.vmax, self.vmax, 
                                           (self.population_size, self._dimension))
        
        self.personal_bests = np.array([np.asarray(sol.representation) for sol in self.population])
        self.personal_best_fitness = np.array([sol.fitness for sol in self.population])
        self.stagnation_counters = np.zeros(self.population_size, dtype=int)
        
        self._update_global_best()
    
    def _update_global_best(self):
        if self.personal_best_fitness is not None:
            best_idx = np.argmin(self.personal_best_fitness)
            if self.personal_best_fitness[best_idx] < self.global_best_fitness:
                self.global_best = self.personal_bests[best_idx].copy()
                self.global_best_fitness = self.personal_best_fitness[best_idx]
        
        if self.global_best is not None:
            self.best_solution = Solution(self.global_best.astype(int).tolist(), self.problem)
            self.best_solution.fitness = self.global_best_fitness
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self.velocities is None:
            self.initialize()
            return
        
        X = np.array([np.asarray(sol.representation, dtype=float) for sol in self.population])
        
        # Velocity update with exploration bias
        r1 = self.rng.random((self.population_size, self._dimension))
        r2 = self.rng.random((self.population_size, self._dimension))
        
        cognitive = self.c1 * r1 * (self.personal_bests - X)
        social = self.c2 * r2 * (self.global_best - X) if self.global_best is not None else 0
        
        self.velocities = self.omega * self.velocities + cognitive + social
        self.velocities = np.clip(self.velocities, -self.vmax, self.vmax)
        
        # Sigmoid probability for binary
        prob = 1.0 / (1.0 + np.exp(-self.velocities))
        
        # Stochastic update (more exploration than deterministic)
        new_X = (self.rng.random((self.population_size, self._dimension)) < prob).astype(int)
        
        # Reinitialize stagnant particles
        for i in range(self.population_size):
            if self.stagnation_counters[i] >= self.reinit_stagnation:
                new_X[i] = self._random_binary(self._dimension)
                self.velocities[i] = self.rng.uniform(-self.vmax, self.vmax, self._dimension)
                self.stagnation_counters[i] = 0
        
        # Create new solutions
        new_population = []
        for i in range(self.population_size):
            sol = Solution(new_X[i].tolist(), self.problem)
            new_population.append(sol)
        
        self.population = new_population
        self.ensure_population_evaluated()
        
        # Update personal bests
        improved = np.zeros(self.population_size, dtype=bool)
        for i in range(self.population_size):
            if self.population[i].fitness < self.personal_best_fitness[i]:
                self.personal_bests[i] = new_X[i].copy()
                self.personal_best_fitness[i] = self.population[i].fitness
                improved[i] = True
                self.stagnation_counters[i] = 0
            else:
                self.stagnation_counters[i] += 1
        
        self._update_global_best()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Genetic Algorithm - Explorer Variant
# =============================================================================

class NKLGAExplorer(BinaryMixin, SearchAlgorithm):
    """
    GA configured for EXPLORATION.
    
    Exploration tuning:
    - High mutation rate (0.15)
    - Uniform crossover (more mixing)
    - Tournament size 2 (low selection pressure)
    - Random immigrants (10% new random individuals each generation)
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        mutation_rate: float = 0.15,      # High mutation
        crossover_rate: float = 0.9,
        tournament_size: int = 2,          # Low selection pressure
        random_immigrant_rate: float = 0.1, # 10% new random each gen
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.random_immigrant_rate = random_immigrant_rate
        self._dimension = None
    
    def initialize(self):
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        new_population = []
        
        # Elitism - keep best
        best = min(self.population, key=lambda s: s.fitness)
        new_population.append(best.copy(preserve_id=False))
        
        # Calculate number of random immigrants
        n_immigrants = int(self.population_size * self.random_immigrant_rate)
        n_offspring = self.population_size - 1 - n_immigrants
        
        # Generate offspring
        while len(new_population) < self.population_size - n_immigrants:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Uniform crossover
            if self.rng.random() < self.crossover_rate:
                child1, child2 = self._uniform_crossover(parent1, parent2)
            else:
                child1 = np.asarray(parent1.representation).copy()
                child2 = np.asarray(parent2.representation).copy()
            
            # Mutation
            child1 = self._bit_flip_mutation(child1, self.mutation_rate)
            child2 = self._bit_flip_mutation(child2, self.mutation_rate)
            
            new_population.append(Solution(child1.tolist(), self.problem))
            if len(new_population) < self.population_size - n_immigrants:
                new_population.append(Solution(child2.tolist(), self.problem))
        
        # Add random immigrants for diversity
        for _ in range(n_immigrants):
            random_ind = self._random_binary(self._dimension)
            new_population.append(Solution(random_ind.tolist(), self.problem))
        
        self.population = new_population[:self.population_size]
        self.ensure_population_evaluated()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1
    
    def _tournament_select(self) -> Solution:
        candidates = self.rng.choice(self.population, size=self.tournament_size, replace=False)
        return min(candidates, key=lambda s: s.fitness)
    
    def _uniform_crossover(self, p1: Solution, p2: Solution) -> Tuple[np.ndarray, np.ndarray]:
        a1 = np.asarray(p1.representation)
        a2 = np.asarray(p2.representation)
        mask = self.rng.random(self._dimension) < 0.5
        c1 = np.where(mask, a1, a2)
        c2 = np.where(mask, a2, a1)
        return c1, c2


# =============================================================================
# Artificial Bee Colony - Explorer Variant
# =============================================================================

class NKLABCExplorer(BinaryMixin, SearchAlgorithm):
    """
    ABC configured for EXPLORATION.
    
    Exploration tuning:
    - High scout limit factor (more scouts = more exploration)
    - High perturbation scale
    - Many onlookers for diverse sampling
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        limit_factor: float = 0.5,        # Lower = more scouts (exploration)
        perturbation_scale: float = 0.8,   # High perturbation
        mutation_rate: float = 0.2,        # Additional mutation
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.limit_factor = limit_factor
        self.perturbation_scale = perturbation_scale
        self.mutation_rate = mutation_rate
        self._dimension = None
        self.trial_counters = []
        self.limit = None
    
    def initialize(self):
        super().initialize()
        if not self.population:
            return
        self._dimension = len(self.population[0].representation)
        self.limit = max(3, int(self.limit_factor * self.population_size * self._dimension))
        self.trial_counters = [0] * self.population_size
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None:
            self.initialize()
            return
        
        # Employed bee phase
        for i in range(self.population_size):
            neighbor = self._generate_neighbor(i)
            if neighbor.fitness <= self.population[i].fitness:
                self.population[i] = neighbor
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
        
        # Calculate selection probabilities
        fitness = np.array([sol.fitness for sol in self.population])
        min_fit = fitness.min()
        adjusted = fitness - min_fit + 1e-10
        probs = 1.0 / adjusted
        probs = probs / probs.sum()
        
        # Onlooker phase
        for _ in range(self.population_size):
            idx = self.rng.choice(self.population_size, p=probs)
            neighbor = self._generate_neighbor(idx)
            if neighbor.fitness <= self.population[idx].fitness:
                self.population[idx] = neighbor
                self.trial_counters[idx] = 0
            else:
                self.trial_counters[idx] += 1
        
        # Scout phase - replace exhausted food sources
        for i in range(self.population_size):
            if self.trial_counters[i] >= self.limit:
                new_binary = self._random_binary(self._dimension)
                self.population[i] = Solution(new_binary.tolist(), self.problem)
                self.population[i].evaluate()
                self.trial_counters[i] = 0
        
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1
    
    def _generate_neighbor(self, idx: int) -> Solution:
        current = np.asarray(self.population[idx].representation)
        
        # Select random partner
        partner_idx = self.rng.integers(self.population_size)
        while partner_idx == idx:
            partner_idx = self.rng.integers(self.population_size)
        partner = np.asarray(self.population[partner_idx].representation)
        
        # Generate neighbor with high perturbation
        n_flip = max(1, int(self._dimension * self.perturbation_scale * self.rng.random()))
        positions = self.rng.choice(self._dimension, size=n_flip, replace=False)
        
        neighbor = current.copy()
        for pos in positions:
            if self.rng.random() < 0.5:
                neighbor[pos] = partner[pos]
            else:
                neighbor[pos] = 1 - neighbor[pos]
        
        # Additional mutation
        neighbor = self._bit_flip_mutation(neighbor, self.mutation_rate)
        
        sol = Solution(neighbor.tolist(), self.problem)
        sol.evaluate()
        return sol


# =============================================================================
# Whale Optimization Algorithm - Explorer Variant
# =============================================================================

class NKLWOAExplorer(BinaryMixin, SearchAlgorithm):
    """
    WOA configured for EXPLORATION.
    
    Exploration tuning:
    - High 'a' values (more |A| > 1 cases = random search)
    - High spiral randomness
    - Lower probability of encircling (p < 0.5)
    - Frequent random whale selection
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        a_initial: float = 3.0,      # Higher a = more |A| > 1 = exploration
        a_final: float = 1.0,        # Don't decay to 0
        spiral_b: float = 2.0,       # Higher b = larger spirals
        encircle_prob: float = 0.3,  # Low encircling probability
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.a_initial = a_initial
        self.a_final = a_final
        self.spiral_b = spiral_b
        self.encircle_prob = encircle_prob
        self.mutation_rate = mutation_rate
        self._dimension = None
        self.max_iterations = kwargs.get('max_iterations', 1000)
    
    def initialize(self):
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        fitness = np.array([sol.fitness for sol in self.population])
        best_idx = np.argmin(fitness)
        best = np.asarray(self.population[best_idx].representation, dtype=float)
        
        progress = min(1.0, self.iteration / max(1, self.max_iterations))
        a = self.a_initial - (self.a_initial - self.a_final) * progress
        
        new_population = []
        for i, whale in enumerate(self.population):
            x = np.asarray(whale.representation, dtype=float)
            
            r1, r2 = self.rng.random(), self.rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = self.rng.random()
            
            if p < self.encircle_prob:
                # Encircling (exploitation-ish, but we minimize this)
                if abs(A) < 1:
                    D = np.abs(C * best - x)
                    new_x = best - A * D
                else:
                    # Random search (exploration)
                    rand_idx = self.rng.integers(self.population_size)
                    rand_whale = np.asarray(self.population[rand_idx].representation, dtype=float)
                    D = np.abs(C * rand_whale - x)
                    new_x = rand_whale - A * D
            else:
                # Spiral with high randomness
                l = self.rng.uniform(-1, 1)
                D = np.abs(best - x)
                new_x = D * np.exp(self.spiral_b * l) * np.cos(2 * np.pi * l) + best
                # Add random perturbation
                new_x += self.rng.normal(0, 0.1, self._dimension)
            
            new_x = np.clip(new_x, 0, 1)
            binary = self._to_binary(new_x)
            binary = self._bit_flip_mutation(binary, self.mutation_rate)
            
            new_population.append(Solution(binary.tolist(), self.problem))
        
        self.population = new_population
        self.ensure_population_evaluated()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Harris Hawks Optimization - Explorer Variant
# =============================================================================

class NKLHHOExplorer(BinaryMixin, SearchAlgorithm):
    """
    HHO configured for EXPLORATION.
    
    Exploration tuning:
    - Force exploration phase (E > 0.5 always)
    - High random hawk following
    - More random mean position influence
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        exploration_bias: float = 0.7,    # Bias toward exploration mode
        random_hawk_prob: float = 0.5,    # High random hawk selection
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.exploration_bias = exploration_bias
        self.random_hawk_prob = random_hawk_prob
        self.mutation_rate = mutation_rate
        self._dimension = None
    
    def initialize(self):
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        positions = np.array([np.asarray(sol.representation, dtype=float) for sol in self.population])
        fitness = np.array([sol.fitness for sol in self.population])
        best_idx = np.argmin(fitness)
        rabbit = positions[best_idx].copy()
        
        new_population = []
        for i in range(self.population_size):
            x = positions[i]
            
            # Force exploration mode by biasing E
            E = self.rng.uniform(0.5, 1.0) if self.rng.random() < self.exploration_bias else self.rng.uniform(-1, 1)
            q = self.rng.random()
            
            # Always exploration step
            if q < self.random_hawk_prob:
                # Follow random hawk
                rand_idx = self.rng.integers(self.population_size)
                rand_hawk = positions[rand_idx]
                new_x = rand_hawk - self.rng.random(self._dimension) * np.abs(rand_hawk - 2 * self.rng.random(self._dimension) * x)
            else:
                # Follow mean position with randomness
                mean_pos = np.mean(positions, axis=0)
                new_x = (rabbit - mean_pos) * self.rng.random(self._dimension) + self.rng.random(self._dimension) * (mean_pos - x)
            
            new_x = np.clip(new_x, 0, 1)
            binary = self._to_binary(new_x)
            binary = self._bit_flip_mutation(binary, self.mutation_rate)
            
            new_population.append(Solution(binary.tolist(), self.problem))
        
        self.population = new_population
        self.ensure_population_evaluated()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Marine Predators Algorithm - Explorer Variant
# =============================================================================

class NKLMPAExplorer(BinaryMixin, SearchAlgorithm):
    """
    MPA configured for EXPLORATION.
    
    Exploration tuning:
    - Force phase 1 (high velocity Brownian)
    - High FAD probability (random jumps)
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        fad_probability: float = 0.4,     # High random jumps
        brownian_scale: float = 1.5,      # Large brownian steps
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.fad_probability = fad_probability
        self.brownian_scale = brownian_scale
        self.mutation_rate = mutation_rate
        self._dimension = None
    
    def initialize(self):
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        positions = np.array([np.asarray(sol.representation, dtype=float) for sol in self.population])
        fitness = np.array([sol.fitness for sol in self.population])
        best_idx = np.argmin(fitness)
        best = positions[best_idx].copy()
        
        new_population = []
        for i in range(self.population_size):
            x = positions[i]
            
            # Phase 1: High velocity Brownian motion (exploration)
            RB = self.rng.normal(0, self.brownian_scale, self._dimension)
            new_x = x + RB * (best - RB * x)
            
            # FAD effect: random jumps for exploration
            if self.rng.random() < self.fad_probability:
                new_x = self.rng.random(self._dimension)
            
            new_x = np.clip(new_x, 0, 1)
            binary = self._to_binary(new_x)
            binary = self._bit_flip_mutation(binary, self.mutation_rate)
            
            new_population.append(Solution(binary.tolist(), self.problem))
        
        self.population = new_population
        self.ensure_population_evaluated()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Slime Mould Algorithm - Explorer Variant
# =============================================================================

class NKLSMAExplorer(BinaryMixin, SearchAlgorithm):
    """
    SMA configured for EXPLORATION.
    
    Exploration tuning:
    - High 'z' equivalent (more random position sampling)
    - Wide weight distribution
    - Random position jumps
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        random_position_prob: float = 0.4,  # High random positioning
        mutation_rate: float = 0.15,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.random_position_prob = random_position_prob
        self.mutation_rate = mutation_rate
        self._dimension = None
    
    def initialize(self):
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        positions = np.array([np.asarray(sol.representation, dtype=float) for sol in self.population])
        fitness = np.array([sol.fitness for sol in self.population])
        idx_sorted = np.argsort(fitness)
        best = positions[idx_sorted[0]]
        
        new_population = []
        for i in range(self.population_size):
            x = positions[i]
            
            if self.rng.random() < self.random_position_prob:
                # Random position for exploration
                new_x = self.rng.random(self._dimension)
            else:
                # Slime movement with high randomness
                weight = self.rng.random(self._dimension) * 2  # Random weights
                j, k = self.rng.choice(self.population_size, size=2, replace=False)
                new_x = best + weight * (positions[j] - positions[k])
            
            new_x = np.clip(new_x, 0, 1)
            binary = self._to_binary(new_x)
            binary = self._bit_flip_mutation(binary, self.mutation_rate)
            
            new_population.append(Solution(binary.tolist(), self.problem))
        
        self.population = new_population
        self.ensure_population_evaluated()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Gravitational Search Algorithm - Explorer Variant
# =============================================================================

class NKLGSAExplorer(BinaryMixin, SearchAlgorithm):
    """
    GSA configured for EXPLORATION.
    
    Exploration tuning:
    - High initial G0 (strong forces)
    - Slow decay (maintain exploration longer)
    - High random component
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        G0: float = 200.0,           # High initial gravitational constant
        alpha: float = 10.0,          # Slow decay
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.G0 = G0
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self._dimension = None
        self.velocities = None
        self.max_iterations = kwargs.get('max_iterations', 1000)
    
    def initialize(self):
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
            self.velocities = np.zeros((self.population_size, self._dimension))
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None:
            self.initialize()
            return
        
        positions = np.array([np.asarray(sol.representation, dtype=float) for sol in self.population])
        fitness = np.array([sol.fitness for sol in self.population])
        
        # Calculate G (slow decay for exploration)
        t = self.iteration / max(1, self.max_iterations)
        G = self.G0 * np.exp(-self.alpha * t)
        
        # Calculate masses
        worst = fitness.max()
        best = fitness.min()
        if np.isclose(worst, best):
            masses = np.ones(self.population_size)
        else:
            masses = (worst - fitness) / (worst - best + 1e-10)
        masses = masses / (masses.sum() + 1e-10)
        
        # Calculate forces and accelerations
        new_population = []
        for i in range(self.population_size):
            force = np.zeros(self._dimension)
            for j in range(self.population_size):
                if i != j:
                    R = np.linalg.norm(positions[i] - positions[j]) + 1e-10
                    direction = positions[j] - positions[i]
                    force += self.rng.random() * G * masses[j] * direction / R
            
            acc = force  # mass cancels
            self.velocities[i] = self.rng.random() * self.velocities[i] + acc
            new_x = positions[i] + self.velocities[i]
            
            new_x = np.clip(new_x, 0, 1)
            binary = self._to_binary(new_x)
            binary = self._bit_flip_mutation(binary, self.mutation_rate)
            
            new_population.append(Solution(binary.tolist(), self.problem))
        
        self.population = new_population
        self.ensure_population_evaluated()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# Diversity Explorer (Pure random with elitism)
# =============================================================================

class NKLDiversityExplorer(BinaryMixin, SearchAlgorithm):
    """
    Pure diversity-focused explorer using random restarts and mutation.
    
    Key properties:
    - High mutation rate
    - Tournament selection with low pressure
    - Random injection to prevent convergence
    """
    phase = "exploration"
    
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        *,
        mutation_rate: float = 0.2,
        random_injection_rate: float = 0.25,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.mutation_rate = mutation_rate
        self.random_injection_rate = random_injection_rate
        self._dimension = None
    
    def initialize(self):
        super().initialize()
        if self.population:
            self._dimension = len(self.population[0].representation)
        self._update_best_solution()
    
    def step(self):
        self.ensure_population_evaluated()
        
        if self._dimension is None and self.population:
            self._dimension = len(self.population[0].representation)
        
        new_population = []
        
        # Keep best (light elitism)
        best = min(self.population, key=lambda s: s.fitness)
        new_population.append(best.copy(preserve_id=False))
        
        for _ in range(self.population_size - 1):
            if self.rng.random() < self.random_injection_rate:
                # Pure random for diversity
                new_x = self._random_binary(self._dimension)
            else:
                # Tournament selection + heavy mutation
                idx1, idx2 = self.rng.choice(len(self.population), 2, replace=False)
                parent = self.population[idx1] if self.population[idx1].fitness < self.population[idx2].fitness else self.population[idx2]
                parent_x = np.asarray(parent.representation)
                new_x = self._bit_flip_mutation(parent_x, self.mutation_rate)
            
            new_population.append(Solution(new_x.tolist(), self.problem))
        
        self.population = new_population
        self.ensure_population_evaluated()
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1
