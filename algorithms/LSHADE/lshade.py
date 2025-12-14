"""
L-SHADE: Success-History based Adaptive Differential Evolution.

References:
- Tanabe, Y., & Fukunaga, A. (2014). Improving the search performance of SHADE using linear population size reduction.
- docs/candidate_L-SHADE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


@dataclass
class LSHADEConfig:
    max_iterations: int = 1000
    min_population: int = 4
    history_size: int = 6
    p_best_rate: float = 0.2


class LSHADE(SearchAlgorithm):
    """Differential evolution with success-history adaptation and population reduction."""

    phase = "exploitation"

    def __init__(
        self,
        problem,
        population_size: int,
        *,
        config: Optional[LSHADEConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(problem, population_size)
        self.config = config or LSHADEConfig()
        self.rng = np.random.default_rng(seed)
        self.dimension = 0
        self.bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.archive: List[np.ndarray] = []
        self.M_F = np.full(self.config.history_size, 0.5)
        self.M_CR = np.full(self.config.history_size, 0.5)
        self.hist_index = 0

    def initialize(self):
        super().initialize()
        self.dimension = self._infer_dimension()
        self.bounds = self._extract_bounds()
        self.archive.clear()
        self.M_F[:] = 0.5
        self.M_CR[:] = 0.5
        self.hist_index = 0
        self.iteration = 0

    def step(self):
        if not self.population:
            self.initialize()
            if not self.population:
                return

        cfg = self.config
        positions = np.array([self._as_vector(sol) for sol in self.population], dtype=float)
        fitness = np.array([sol.fitness if sol.fitness is not None else sol.evaluate() for sol in self.population], dtype=float)
        pop_size = positions.shape[0]
        next_positions = positions.copy()
        next_solutions = [sol.copy(preserve_id=False) for sol in self.population]
        SF = []
        SCR = []
        delta_f = []

        for i in range(pop_size):
            Fi, CRi = self._sample_parameters()
            donor = self._mutate(i, Fi, positions, fitness)
            trial = self._crossover(positions[i], donor, CRi)
            trial = self._clip(trial)
            trial_solution = Solution(trial, self.problem)
            trial_fit = trial_solution.evaluate()
            if trial_fit <= fitness[i]:
                next_positions[i] = trial
                next_solutions[i] = trial_solution
                SF.append(Fi)
                SCR.append(CRi)
                delta_f.append(abs(trial_fit - fitness[i]))
                self.archive.append(positions[i])
            else:
                next_positions[i] = positions[i]
                next_solutions[i] = self.population[i]

        if SF:
            self._update_history(SF, SCR, delta_f)

        # Linear population reduction
        target_size = self._current_population_size()
        if next_positions.shape[0] > target_size:
            sorted_idx = np.argsort([sol.fitness if sol.fitness is not None else sol.evaluate() for sol in next_solutions])
            keep_idx = sorted_idx[:target_size]
            next_positions = next_positions[keep_idx]
            next_solutions = [next_solutions[idx] for idx in keep_idx]
        self.population = [sol.copy(preserve_id=False) for sol in next_solutions]
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()
        self.iteration += 1

    # --- Mutation and crossover --------------------------------------------
    def _mutate(self, idx, F, positions, fitness):
        pop_size = positions.shape[0]
        p_num = max(2, int(np.ceil(self.config.p_best_rate * pop_size)))
        p_best_indices = np.argsort(fitness)[:p_num]
        p_best = positions[self.rng.choice(p_best_indices)]
        idxs = list(range(pop_size))
        idxs.remove(idx)
        r1, r2 = self.rng.choice(idxs, size=2, replace=False)
        x_a = positions[r1]
        x_b = self._select_from_population_or_archive(r2, positions)
        mutant = positions[idx] + F * (p_best - positions[idx]) + F * (x_a - x_b)
        return mutant

    def _crossover(self, target, donor, CR):
        mask = self.rng.random(self.dimension) < CR
        j_rand = self.rng.integers(self.dimension)
        mask[j_rand] = True
        trial = np.where(mask, donor, target)
        return trial

    # --- Adaptation --------------------------------------------------------
    def _sample_parameters(self):
        k = self.rng.integers(self.config.history_size)
        mu_F = self.M_F[k]
        mu_CR = self.M_CR[k]
        Fi = self._cauchy(mu_F, 0.1)
        while Fi <= 0:
            Fi = self._cauchy(mu_F, 0.1)
        Fi = min(Fi, 1.0)
        CRi = self.rng.normal(mu_CR, 0.1)
        CRi = np.clip(CRi, 0.0, 1.0)
        return Fi, CRi

    def _update_history(self, SF, SCR, delta_f):
        weights = np.array(delta_f)
        if np.sum(weights) == 0:
            return
        weights /= np.sum(weights)
        SF = np.array(SF)
        SCR = np.array(SCR)
        mean_F = np.sum(weights * (SF ** 2)) / np.sum(weights * SF)
        mean_CR = np.sum(weights * SCR)
        self.M_F[self.hist_index] = mean_F
        self.M_CR[self.hist_index] = mean_CR
        self.hist_index = (self.hist_index + 1) % self.config.history_size

    # --- Helpers -----------------------------------------------------------
    def _current_population_size(self):
        cfg = self.config
        init = self.population_size
        min_pop = max(4, cfg.min_population)
        t = min(self.iteration, cfg.max_iterations)
        size = round(((min_pop - init) / max(1, cfg.max_iterations)) * t + init)
        return max(min_pop, size)

    def _select_from_population_or_archive(self, index, positions):
        if self.archive and self.rng.random() < 0.5:
            idx = self.rng.integers(len(self.archive))
            return self.archive[idx]
        return positions[index]

    def _cauchy(self, loc, scale):
        return loc + scale * np.tan(np.pi * (self.rng.random() - 0.5))

    def _infer_dimension(self):
        if not self.population:
            return 0
        rep = self.population[0].representation
        if hasattr(rep, "__len__"):
            return len(rep)
        return 1

    def _extract_bounds(self):
        try:
            info = self.problem.get_problem_info()
        except Exception:
            info = {}
        lowers = np.asarray(info.get("lower_bounds")) if "lower_bounds" in info else None
        uppers = np.asarray(info.get("upper_bounds")) if "upper_bounds" in info else None
        if lowers is None and uppers is None:
            return None
        if lowers is None:
            lowers = np.full(self.dimension, -np.inf)
        if uppers is None:
            uppers = np.full(self.dimension, np.inf)
        if lowers.size == 1:
            lowers = np.full(self.dimension, lowers.item())
        if uppers.size == 1:
            uppers = np.full(self.dimension, uppers.item())
        return lowers.astype(float), uppers.astype(float)

    def _clip(self, vector):
        if self.bounds is None:
            return vector
        lower, upper = self.bounds
        return np.clip(vector, lower, upper)

    def _as_vector(self, solution: Solution):
        arr = np.asarray(solution.representation, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr
