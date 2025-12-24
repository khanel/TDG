"""
MaxCut Explorer Variants - Configured for DIVERSITY and GLOBAL SEARCH.

11 explorer variants tuned for:
- High mutation rates / perturbation
- Weak selection pressure
- Diversity maintenance
- Random injection / immigration
- Broad parameter exploration

Binary domain with graph-cut structure.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


class MaxCutBinaryMixin:
    """
    Utility mixin for MaxCut binary operations.
    Converts continuous vectors to binary partition masks.
    """

    def _binarize(self, vector: np.ndarray) -> np.ndarray:
        """Convert continuous values to binary using sigmoid threshold."""
        return (vector >= 0.5).astype(int)

    def _make_solution(self, mask: np.ndarray) -> Solution:
        """Create and evaluate a solution from binary mask."""
        sol = Solution(mask.astype(int).tolist(), self.problem)
        sol.evaluate()
        return sol

    def _random_binary(self, size: int) -> np.ndarray:
        """Generate random binary vector."""
        return self.rng.integers(0, 2, size=size)

    @staticmethod
    def _ensure_evaluated(population: List[Solution]) -> None:
        for sol in population:
            if sol.fitness is None:
                sol.evaluate()


# =============================================================================
# 1. MAP-Elites Explorer - Quality-Diversity
# =============================================================================
class MaxCutMapElitesExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    MAP-Elites style diversity explorer for MaxCut.
    
    Maintains diverse archive based on behavioral descriptors
    (partition balance, cluster count).
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 64,
        *,
        n_bins: int = 10,
        mutation_rate: float = 0.15,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutMapElitesExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.n_bins = n_bins
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        self.archive: dict = {}  # (bin_x, bin_y) -> Solution

    def _get_behavior(self, solution: Solution) -> tuple:
        """Extract behavioral descriptor: partition balance."""
        mask = np.asarray(solution.representation, dtype=int)
        balance = np.mean(mask)  # 0 to 1
        # Cluster diversity (simplified)
        transitions = np.sum(np.abs(np.diff(mask)))
        diversity = transitions / max(1, len(mask) - 1)
        bin_x = min(int(balance * self.n_bins), self.n_bins - 1)
        bin_y = min(int(diversity * self.n_bins), self.n_bins - 1)
        return (bin_x, bin_y)

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)
        for sol in self.population:
            self._try_add_to_archive(sol)

    def _try_add_to_archive(self, solution: Solution):
        """Add solution to archive if it improves its niche."""
        key = self._get_behavior(solution)
        if key not in self.archive or (
            solution.fitness is not None
            and (self.archive[key].fitness is None or solution.fitness < self.archive[key].fitness)
        ):
            self.archive[key] = solution.copy(preserve_id=False)

    def step(self):
        self._ensure_evaluated(self.population)

        # Sample parents from archive
        archive_list = list(self.archive.values())
        if not archive_list:
            archive_list = self.population

        offspring: List[Solution] = []
        for _ in range(self.population_size):
            parent = self.rng.choice(archive_list)
            mask = np.asarray(parent.representation, dtype=int).copy()

            # High mutation for exploration
            flip_mask = self.rng.random(len(mask)) < self.mutation_rate
            if not np.any(flip_mask):
                flip_mask[self.rng.integers(len(mask))] = True
            mask[flip_mask] = 1 - mask[flip_mask]

            child = self._make_solution(mask)
            offspring.append(child)
            self._try_add_to_archive(child)

        # Population from archive + best offspring
        archive_list = list(self.archive.values())
        combined = archive_list + offspring
        combined.sort(key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        self.population = combined[: self.population_size]

        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 2. GA Explorer - Genetic Algorithm (exploration-tuned)
# =============================================================================
class MaxCutGAExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    GA explorer tuned for diversity:
    - High mutation rate
    - Weak selection pressure
    - Random immigrant injection
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 64,
        *,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        tournament_size: int = 2,
        random_injection_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutGAExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.random_injection_rate = random_injection_rate
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)

    def _tournament_select(self) -> Solution:
        """Weak tournament selection."""
        candidates = self.rng.choice(self.population, size=self.tournament_size, replace=False)
        return min(candidates, key=lambda s: s.fitness if s.fitness is not None else float("inf"))

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        offspring: List[Solution] = []
        while len(offspring) < self.population_size:
            # Random immigrant injection
            if self.rng.random() < self.random_injection_rate:
                mask = self._random_binary(dim)
                offspring.append(self._make_solution(mask))
                continue

            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            mask1 = np.asarray(parent1.representation, dtype=int)
            mask2 = np.asarray(parent2.representation, dtype=int)

            # Uniform crossover
            if self.rng.random() < self.crossover_rate:
                swap = self.rng.random(dim) < 0.5
                child_mask = np.where(swap, mask2, mask1)
            else:
                child_mask = mask1.copy()

            # High mutation
            flip = self.rng.random(dim) < self.mutation_rate
            if not np.any(flip):
                flip[self.rng.integers(dim)] = True
            child_mask[flip] = 1 - child_mask[flip]

            offspring.append(self._make_solution(child_mask))

        combined = self.population + offspring
        combined.sort(key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        self.population = combined[: self.population_size]

        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 3. PSO Explorer - Particle Swarm Optimization (exploration-tuned)
# =============================================================================
class MaxCutPSOExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    Binary PSO explorer tuned for exploration:
    - High inertia (0.9)
    - Strong cognitive (personal best attraction)
    - Weak social (global best attraction)
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 56,
        *,
        w: float = 0.9,
        c1: float = 2.5,
        c2: float = 0.5,
        v_max: float = 4.0,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutPSOExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.rng = np.random.default_rng(seed)
        self.velocities: Optional[np.ndarray] = None
        self.pbest: Optional[List[Solution]] = None
        self.gbest: Optional[Solution] = None

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)
        self.velocities = self.rng.uniform(-self.v_max, self.v_max, (self.population_size, dim))
        self.pbest = [sol.copy(preserve_id=False) for sol in self.population]
        self._update_gbest()

    def _update_gbest(self):
        best = min(self.population, key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        if self.gbest is None or (best.fitness is not None and best.fitness < (self.gbest.fitness or float("inf"))):
            self.gbest = best.copy(preserve_id=False)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        new_population: List[Solution] = []
        for i, sol in enumerate(self.population):
            pos = np.asarray(sol.representation, dtype=float)
            pbest_pos = np.asarray(self.pbest[i].representation, dtype=float)
            gbest_pos = np.asarray(self.gbest.representation, dtype=float)

            r1, r2 = self.rng.random(dim), self.rng.random(dim)
            cognitive = self.c1 * r1 * (pbest_pos - pos)
            social = self.c2 * r2 * (gbest_pos - pos)

            self.velocities[i] = self.w * self.velocities[i] + cognitive + social
            self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

            # Binary conversion via sigmoid
            prob = self._sigmoid(self.velocities[i])
            new_mask = (self.rng.random(dim) < prob).astype(int)

            child = self._make_solution(new_mask)
            new_population.append(child)

            # Update personal best
            if child.fitness is not None and (
                self.pbest[i].fitness is None or child.fitness < self.pbest[i].fitness
            ):
                self.pbest[i] = child.copy(preserve_id=False)

        self.population = new_population
        self._update_gbest()
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 4. GWO Explorer - Grey Wolf Optimizer (exploration-tuned)
# =============================================================================
class MaxCutGWOExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    GWO explorer tuned for exploration:
    - Slow a decay (stays in exploration longer)
    - Random wolf injection
    - High perturbation
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 48,
        *,
        a_initial: float = 3.0,
        a_final: float = 1.0,
        random_wolf_prob: float = 0.3,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutGWOExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.a_initial = a_initial
        self.a_final = a_final
        self.random_wolf_prob = random_wolf_prob
        self.rng = np.random.default_rng(seed)
        self.max_iterations = 1000

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)

    def _get_leaders(self) -> tuple:
        """Get alpha, beta, delta wolves."""
        sorted_pop = sorted(
            self.population,
            key=lambda s: s.fitness if s.fitness is not None else float("inf"),
        )
        alpha = np.asarray(sorted_pop[0].representation, dtype=float)
        beta = np.asarray(sorted_pop[1].representation, dtype=float) if len(sorted_pop) > 1 else alpha
        delta = np.asarray(sorted_pop[2].representation, dtype=float) if len(sorted_pop) > 2 else beta
        return alpha, beta, delta

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        # Slow decay for exploration
        progress = min(self.iteration / self.max_iterations, 1.0)
        a = self.a_initial - progress * (self.a_initial - self.a_final)

        alpha, beta, delta = self._get_leaders()

        new_population: List[Solution] = []
        for sol in self.population:
            # Random wolf injection
            if self.rng.random() < self.random_wolf_prob:
                new_mask = self._random_binary(dim)
                new_population.append(self._make_solution(new_mask))
                continue

            pos = np.asarray(sol.representation, dtype=float)

            # GWO update equations
            A1, A2, A3 = 2 * a * self.rng.random(dim) - a, 2 * a * self.rng.random(dim) - a, 2 * a * self.rng.random(dim) - a
            C1, C2, C3 = 2 * self.rng.random(dim), 2 * self.rng.random(dim), 2 * self.rng.random(dim)

            D_alpha = np.abs(C1 * alpha - pos)
            D_beta = np.abs(C2 * beta - pos)
            D_delta = np.abs(C3 * delta - pos)

            X1 = alpha - A1 * D_alpha
            X2 = beta - A2 * D_beta
            X3 = delta - A3 * D_delta

            new_pos = (X1 + X2 + X3) / 3.0
            new_mask = self._binarize(new_pos)

            new_population.append(self._make_solution(new_mask))

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 5. ABC Explorer - Artificial Bee Colony (exploration-tuned)
# =============================================================================
class MaxCutABCExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    ABC explorer tuned for exploration:
    - Low limit (quick scout deployment)
    - High perturbation
    - More scout bees
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 72,
        *,
        limit_factor: float = 0.5,
        perturbation_scale: float = 0.8,
        scout_rate: float = 0.2,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutABCExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.limit_factor = limit_factor
        self.perturbation_scale = perturbation_scale
        self.scout_rate = scout_rate
        self.rng = np.random.default_rng(seed)
        self.trials: Optional[np.ndarray] = None
        self.limit: int = 1

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)
        self.limit = max(1, int(self.limit_factor * dim))
        self.trials = np.zeros(self.population_size, dtype=int)

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        # Employed bee phase
        for i, sol in enumerate(self.population):
            mask = np.asarray(sol.representation, dtype=int).copy()

            # Select random partner
            k = self.rng.choice([j for j in range(self.population_size) if j != i])
            partner_mask = np.asarray(self.population[k].representation, dtype=int)

            # Perturb multiple positions
            n_flips = max(1, int(self.perturbation_scale * dim * self.rng.random()))
            flip_indices = self.rng.choice(dim, size=min(n_flips, dim), replace=False)
            
            for idx in flip_indices:
                if self.rng.random() < 0.5:
                    mask[idx] = partner_mask[idx]
                else:
                    mask[idx] = 1 - mask[idx]

            candidate = self._make_solution(mask)
            if candidate.fitness is not None and (sol.fitness is None or candidate.fitness < sol.fitness):
                self.population[i] = candidate
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        # Scout bee phase
        for i in range(self.population_size):
            if self.trials[i] >= self.limit or self.rng.random() < self.scout_rate:
                new_mask = self._random_binary(dim)
                self.population[i] = self._make_solution(new_mask)
                self.trials[i] = 0

        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 6. WOA Explorer - Whale Optimization Algorithm (exploration-tuned)
# =============================================================================
class MaxCutWOAExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    WOA explorer tuned for exploration:
    - High initial 'a' parameter
    - More random search
    - Slow decay
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 48,
        *,
        a_initial: float = 3.0,
        b: float = 1.0,
        encircle_prob: float = 0.3,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutWOAExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.a_initial = a_initial
        self.b = b
        self.encircle_prob = encircle_prob
        self.rng = np.random.default_rng(seed)
        self.max_iterations = 1000

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        # Slow decay
        progress = min(self.iteration / self.max_iterations, 1.0)
        a = self.a_initial - progress * self.a_initial * 0.5  # Only decay to half

        best = min(self.population, key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        best_pos = np.asarray(best.representation, dtype=float)

        new_population: List[Solution] = []
        for sol in self.population:
            pos = np.asarray(sol.representation, dtype=float)
            A = 2 * a * self.rng.random(dim) - a
            C = 2 * self.rng.random(dim)

            if self.rng.random() < self.encircle_prob:
                # Encircling prey (exploitation-like but with high randomness)
                D = np.abs(C * best_pos - pos)
                new_pos = best_pos - A * D
            elif np.abs(A).mean() >= 1:
                # Random search (exploration)
                rand_idx = self.rng.integers(self.population_size)
                rand_pos = np.asarray(self.population[rand_idx].representation, dtype=float)
                D = np.abs(C * rand_pos - pos)
                new_pos = rand_pos - A * D
            else:
                # Spiral update
                l = self.rng.uniform(-1, 1, dim)
                D = np.abs(best_pos - pos)
                new_pos = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_pos

            new_mask = self._binarize(new_pos)
            new_population.append(self._make_solution(new_mask))

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 7. HHO Explorer - Harris Hawks Optimization (exploration-tuned)
# =============================================================================
class MaxCutHHOExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    HHO explorer tuned for exploration:
    - High exploration bias
    - More random hawk behavior
    - Large jumps
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 56,
        *,
        exploration_bias: float = 0.7,
        random_hawk_prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutHHOExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.exploration_bias = exploration_bias
        self.random_hawk_prob = random_hawk_prob
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        rabbit = min(self.population, key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        rabbit_pos = np.asarray(rabbit.representation, dtype=float)

        # Mean position for exploration
        mean_pos = np.mean([np.asarray(s.representation, dtype=float) for s in self.population], axis=0)

        new_population: List[Solution] = []
        for sol in self.population:
            pos = np.asarray(sol.representation, dtype=float)
            E = 2 * (1 - self.rng.random())  # Escaping energy

            if self.rng.random() < self.exploration_bias or abs(E) >= 1:
                # Exploration phase
                if self.rng.random() < self.random_hawk_prob:
                    # Random position
                    new_pos = self.rng.random(dim)
                else:
                    # Move based on random hawk and mean
                    rand_idx = self.rng.integers(self.population_size)
                    rand_pos = np.asarray(self.population[rand_idx].representation, dtype=float)
                    q = self.rng.random()
                    new_pos = rand_pos - self.rng.random(dim) * np.abs(rand_pos - 2 * self.rng.random(dim) * pos)
            else:
                # Soft besiege with random jumps
                J = 2 * (1 - self.rng.random(dim))
                new_pos = rabbit_pos - pos - E * np.abs(J * rabbit_pos - pos)
                # Add random perturbation
                new_pos += self.rng.uniform(-0.3, 0.3, dim)

            new_mask = self._binarize(new_pos)
            new_population.append(self._make_solution(new_mask))

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 8. MPA Explorer - Marine Predators Algorithm (exploration-tuned)
# =============================================================================
class MaxCutMPAExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    MPA explorer tuned for exploration:
    - High FAD probability
    - Large Brownian scale
    - More random movements
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 60,
        *,
        fad_probability: float = 0.4,
        brownian_scale: float = 1.5,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutMPAExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.fad_probability = fad_probability
        self.brownian_scale = brownian_scale
        self.rng = np.random.default_rng(seed)
        self.max_iterations = 1000

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)

    def _levy_flight(self, dim: int) -> np.ndarray:
        """Generate Levy flight step."""
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = self.rng.normal(0, sigma, dim)
        v = self.rng.normal(0, 1, dim)
        return u / (np.abs(v) ** (1 / beta))

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        progress = min(self.iteration / self.max_iterations, 1.0)
        elite = min(self.population, key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        elite_pos = np.asarray(elite.representation, dtype=float)

        new_population: List[Solution] = []
        for sol in self.population:
            pos = np.asarray(sol.representation, dtype=float)

            if progress < 0.5:
                # Phase 1: High exploration (Brownian motion)
                R_B = self.rng.normal(0, 1, dim) * self.brownian_scale
                stepsize = R_B * (elite_pos - R_B * pos)
                new_pos = pos + stepsize
            else:
                # Mixed phase with Levy flights
                R_L = self._levy_flight(dim)
                if self.rng.random() < 0.5:
                    stepsize = R_L * (elite_pos - R_L * pos)
                else:
                    CF = (1 - progress) ** 2
                    R_B = self.rng.normal(0, 1, dim) * self.brownian_scale
                    stepsize = CF * R_B * (elite_pos - CF * pos)
                new_pos = pos + stepsize

            # FAD effect (high probability)
            if self.rng.random() < self.fad_probability:
                U = self.rng.random(dim) < self.fad_probability
                new_pos = new_pos + U * self.rng.random(dim)

            new_mask = self._binarize(new_pos)
            new_population.append(self._make_solution(new_mask))

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 9. SMA Explorer - Slime Mould Algorithm (exploration-tuned)
# =============================================================================
class MaxCutSMAExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    SMA explorer tuned for exploration:
    - High random position probability
    - High mutation rate
    - More oscillation
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 64,
        *,
        random_position_prob: float = 0.4,
        z: float = 0.03,
        mutation_rate: float = 0.15,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutSMAExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.random_position_prob = random_position_prob
        self.z = z
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        self.max_iterations = 1000

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        progress = min(self.iteration / self.max_iterations, 1.0)
        eps = 1e-12
        x = 1.0 - float(progress)
        x = max(-1.0 + eps, min(1.0 - eps, x))
        a = np.arctanh(x)  # Decreases over time
        b = 1 - progress

        best = min(self.population, key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        best_pos = np.asarray(best.representation, dtype=float)

        # Sort by fitness for weight calculation
        sorted_pop = sorted(
            self.population,
            key=lambda s: s.fitness if s.fitness is not None else float("inf"),
        )
        best_fit = float(best.fitness) if best.fitness is not None else 0.0
        weights = np.zeros(self.population_size)
        for i, sol in enumerate(sorted_pop):
            sol_fit = float(sol.fitness) if sol.fitness is not None else best_fit
            ratio = (best_fit + eps) / (sol_fit + eps) + 1.0
            ratio = max(ratio, eps)
            log_term = math.log10(ratio)
            if i < self.population_size // 2:
                weights[i] = 1 + self.rng.random() * log_term
            else:
                weights[i] = 1 - self.rng.random() * log_term

        new_population: List[Solution] = []
        for i, sol in enumerate(self.population):
            pos = np.asarray(sol.representation, dtype=float)

            if self.rng.random() < self.random_position_prob:
                # Random exploration
                new_pos = self.rng.random(dim)
            else:
                sol_fit = float(sol.fitness) if sol.fitness is not None else best_fit
                p = np.tanh(np.abs(sol_fit - best_fit) + eps)
                vb = 2 * a * self.rng.random(dim) - a
                vc = 2 * b * self.rng.random(dim) - b

                if self.rng.random() < p:
                    # Random individuals for exploration
                    rand_a = sorted_pop[self.rng.integers(self.population_size)]
                    rand_b = sorted_pop[self.rng.integers(self.population_size)]
                    pos_a = np.asarray(rand_a.representation, dtype=float)
                    pos_b = np.asarray(rand_b.representation, dtype=float)
                    new_pos = best_pos + vb * (weights[i] * pos_a - pos_b)
                else:
                    new_pos = vc * pos

            # Additional mutation for diversity
            if self.rng.random() < self.mutation_rate:
                flip_mask = self.rng.random(dim) < self.mutation_rate
                bin_pos = self._binarize(new_pos)
                bin_pos[flip_mask] = 1 - bin_pos[flip_mask]
                new_mask = bin_pos
            else:
                new_mask = self._binarize(new_pos)

            new_population.append(self._make_solution(new_mask))

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 10. GSA Explorer - Gravitational Search Algorithm (exploration-tuned)
# =============================================================================
class MaxCutGSAExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    GSA explorer tuned for exploration:
    - High initial G0
    - Low alpha (slow decay)
    - High k_best ratio
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 56,
        *,
        G0: float = 200.0,
        alpha: float = 10.0,
        k_best_ratio: float = 0.6,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutGSAExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.G0 = G0
        self.alpha = alpha
        self.k_best_ratio = k_best_ratio
        self.rng = np.random.default_rng(seed)
        self.velocities: Optional[np.ndarray] = None
        self.max_iterations = 1000

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)
        self.velocities = np.zeros((self.population_size, dim))

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        progress = min(self.iteration / self.max_iterations, 1.0)
        G = self.G0 * np.exp(-self.alpha * progress)

        # Calculate masses
        fitnesses = np.array([s.fitness if s.fitness is not None else float("inf") for s in self.population])
        worst = np.max(fitnesses[fitnesses < float("inf")]) if np.any(fitnesses < float("inf")) else 1.0
        best_fit = np.min(fitnesses)
        
        masses = np.zeros(self.population_size)
        for i, fit in enumerate(fitnesses):
            if fit < float("inf"):
                masses[i] = (worst - fit) / (worst - best_fit + 1e-10)
        masses = masses / (np.sum(masses) + 1e-10)

        # K best agents
        k = max(1, int(self.k_best_ratio * self.population_size * (1 - progress * 0.5)))
        sorted_indices = np.argsort(fitnesses)[:k]

        # Calculate forces
        positions = np.array([np.asarray(s.representation, dtype=float) for s in self.population])
        forces = np.zeros((self.population_size, dim))

        for i in range(self.population_size):
            for j in sorted_indices:
                if i != j:
                    R = np.linalg.norm(positions[i] - positions[j]) + 1e-10
                    force_mag = G * masses[i] * masses[j] / R
                    forces[i] += self.rng.random(dim) * force_mag * (positions[j] - positions[i])

        # Update velocities and positions
        new_population: List[Solution] = []
        for i in range(self.population_size):
            if masses[i] > 1e-10:
                acc = forces[i] / masses[i]
            else:
                acc = forces[i]
            
            self.velocities[i] = self.rng.random(dim) * self.velocities[i] + acc
            new_pos = positions[i] + self.velocities[i]
            new_mask = self._binarize(new_pos)
            new_population.append(self._make_solution(new_mask))

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1


# =============================================================================
# 11. Diversity Explorer - Pure diversity maintenance
# =============================================================================
class MaxCutDiversityExplorer(MaxCutBinaryMixin, SearchAlgorithm):
    """
    Pure diversity explorer:
    - Maximizes population diversity
    - High mutation rate
    - Random injection
    - Novelty-based selection
    """
    phase = "exploration"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 48,
        *,
        mutation_rate: float = 0.2,
        random_injection_rate: float = 0.25,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "maxcut_problem"):
            raise ValueError("MaxCutDiversityExplorer expects a MaxCutAdapter.")
        super().__init__(problem, population_size)
        self.mutation_rate = mutation_rate
        self.random_injection_rate = random_injection_rate
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        super().initialize()
        self._ensure_evaluated(self.population)

    def _hamming_distance(self, sol1: Solution, sol2: Solution) -> float:
        """Calculate normalized Hamming distance between two solutions."""
        m1 = np.asarray(sol1.representation, dtype=int)
        m2 = np.asarray(sol2.representation, dtype=int)
        return np.sum(m1 != m2) / len(m1)

    def _novelty_score(self, sol: Solution, archive: List[Solution], k: int = 5) -> float:
        """Calculate novelty score based on k-nearest neighbors."""
        if not archive:
            return 1.0
        distances = [self._hamming_distance(sol, other) for other in archive]
        distances.sort(reverse=True)
        return np.mean(distances[: min(k, len(distances))])

    def step(self):
        self._ensure_evaluated(self.population)
        dim = len(self.population[0].representation)

        offspring: List[Solution] = []
        for _ in range(self.population_size):
            # Random injection
            if self.rng.random() < self.random_injection_rate:
                mask = self._random_binary(dim)
                offspring.append(self._make_solution(mask))
                continue

            # Select parent with preference for novel solutions
            novelty_scores = [self._novelty_score(sol, self.population) for sol in self.population]
            probs = np.array(novelty_scores) / (np.sum(novelty_scores) + 1e-10)
            parent_idx = self.rng.choice(len(self.population), p=probs)
            parent = self.population[parent_idx]

            mask = np.asarray(parent.representation, dtype=int).copy()

            # High mutation
            flip = self.rng.random(dim) < self.mutation_rate
            if not np.any(flip):
                n_flips = max(1, int(self.mutation_rate * dim))
                flip_indices = self.rng.choice(dim, size=n_flips, replace=False)
                flip[flip_indices] = True
            mask[flip] = 1 - mask[flip]

            offspring.append(self._make_solution(mask))

        # Select based on combination of fitness and novelty
        combined = self.population + offspring
        scores = []
        for sol in combined:
            fit_score = sol.fitness if sol.fitness is not None else float("inf")
            nov_score = self._novelty_score(sol, combined)
            # Weight novelty more for exploration
            combined_score = 0.4 * fit_score - 0.6 * nov_score * 1000
            scores.append((combined_score, sol))

        scores.sort(key=lambda x: x[0])
        self.population = [s[1] for s in scores[: self.population_size]]

        self._update_best_solution()
        self.iteration += 1


__all__ = [
    "MaxCutMapElitesExplorer",
    "MaxCutGAExplorer",
    "MaxCutPSOExplorer",
    "MaxCutGWOExplorer",
    "MaxCutABCExplorer",
    "MaxCutWOAExplorer",
    "MaxCutHHOExplorer",
    "MaxCutMPAExplorer",
    "MaxCutSMAExplorer",
    "MaxCutGSAExplorer",
    "MaxCutDiversityExplorer",
]
