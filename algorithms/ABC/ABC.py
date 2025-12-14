"""
Artificial Bee Colony (ABC) search algorithm.

The implementation follows the canonical employed/onlooker/scout phases
described by Karaboga (2005) with additional handling for discrete and
permutation representations so it can plug directly into the project's
problem abstractions.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


class ArtificialBeeColony(SearchAlgorithm):
    """Population-based ABC implementation with generic neighbor support."""

    phase = "exploration"

    def __init__(
        self,
        problem,
        population_size: int,
        *,
        onlooker_count: Optional[int] = None,
        limit: Optional[int] = None,
        limit_factor: float = 1.0,
        perturbation_scale: float = 0.5,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.onlooker_count = int(onlooker_count) if onlooker_count else population_size
        self.limit = int(limit) if limit is not None else None
        self.limit_factor = float(max(0.1, limit_factor))
        self.perturbation_scale = float(max(1e-6, perturbation_scale))
        self.trial_counters: List[int] = []
        self._problem_type = "continuous"
        self._lower_bounds: Optional[np.ndarray] = None
        self._upper_bounds: Optional[np.ndarray] = None
        self._dimension: int = 0

    def initialize(self):
        """Initialize food sources and counters."""
        super().initialize()
        if not self.population:
            return
        info = {}
        try:
            info = self.problem.get_problem_info()
        except Exception:
            info = {}
        self._problem_type = str(info.get("problem_type", "continuous")).lower()
        self._dimension = int(info.get("dimension") or self._infer_dimension(self.population[0]))
        self._lower_bounds = self._as_array(info.get("lower_bounds"))
        self._upper_bounds = self._as_array(info.get("upper_bounds"))
        if self.onlooker_count <= 0:
            self.onlooker_count = self.population_size
        if self.limit is None:
            approx_dim = max(1, self._dimension)
            derived = int(self.limit_factor * self.population_size * approx_dim)
            self.limit = max(5, derived)
        self.trial_counters = [0 for _ in range(len(self.population))]
        self.iteration = 0
        self._update_best_solution()

    def ingest_population(self, seeds: List[Solution]) -> None:
        super().ingest_population(seeds)
        if not self.population:
            return
        if len(self.trial_counters) != len(self.population):
            self.trial_counters = [0 for _ in range(len(self.population))]

    def step(self):
        if not self.population:
            self.initialize()
            if not self.population:
                return
        self._employed_phase()
        probabilities = self._calculate_probabilities()
        self._onlooker_phase(probabilities)
        self._scout_phase()
        self._update_best_solution()
        self.iteration += 1

    # --- Phases -------------------------------------------------------------
    def _employed_phase(self) -> None:
        for idx in range(len(self.population)):
            candidate = self._generate_neighbor(idx)
            self._greedy_select(idx, candidate)

    def _onlooker_phase(self, probabilities: Sequence[float]) -> None:
        if not probabilities:
            return
        for _ in range(self.onlooker_count):
            idx = self._select_index(probabilities)
            if idx is None:
                break
            candidate = self._generate_neighbor(idx)
            self._greedy_select(idx, candidate)

    def _scout_phase(self) -> None:
        if self.limit is None or self.limit <= 0:
            return
        for idx, trials in enumerate(self.trial_counters):
            if trials < self.limit:
                continue
            scout = self.problem.get_initial_solution()
            scout.evaluate()
            self.population[idx] = scout
            self.trial_counters[idx] = 0

    # --- Helpers ------------------------------------------------------------
    def _calculate_probabilities(self) -> List[float]:
        fitness = []
        for sol in self.population:
            fit = sol.fitness if sol.fitness is not None else sol.evaluate()
            if not np.isfinite(fit):
                fit = float("inf")
            fitness.append(fit)
        if not fitness:
            return []
        min_fit = min(fitness)
        shift = -min_fit + 1e-9 if min_fit < 0 else 0.0
        qualities = [1.0 / (1.0 + f + shift) for f in fitness]
        total = float(sum(qualities))
        if not np.isfinite(total) or total <= 0:
            return [1.0 / len(fitness)] * len(fitness)
        return [q / total for q in qualities]

    def _select_index(self, probabilities: Sequence[float]) -> Optional[int]:
        if not probabilities:
            return None
        cumulative = np.cumsum(probabilities)
        r = self.rng.random()
        for idx, threshold in enumerate(cumulative):
            if r <= threshold:
                return idx
        return len(probabilities) - 1

    def _greedy_select(self, index: int, candidate: Solution) -> None:
        current = self.population[index]
        new_fit = candidate.evaluate()
        current_fit = current.fitness if current.fitness is not None else current.evaluate()
        if new_fit < current_fit:
            self.population[index] = candidate
            self.trial_counters[index] = 0
        else:
            self.trial_counters[index] += 1

    def _generate_neighbor(self, index: int) -> Solution:
        if len(self.population) == 1:
            base = self.population[index].copy(preserve_id=False)
            base.evaluate()
            return base
        partner_idx = self._pick_partner(index)
        if self._problem_type in {"continuous", "real"}:
            return self._continuous_neighbor(index, partner_idx)
        return self._discrete_neighbor(index, partner_idx)

    def _continuous_neighbor(self, index: int, partner_idx: int) -> Solution:
        origin = np.asarray(self.population[index].representation, dtype=float)
        partner = np.asarray(self.population[partner_idx].representation, dtype=float)
        phi = self.rng.uniform(-self.perturbation_scale, self.perturbation_scale, size=origin.shape)
        candidate = origin + phi * (origin - partner)
        candidate = self._clip(candidate)
        return Solution(candidate, self.problem)

    def _discrete_neighbor(self, index: int, partner_idx: int) -> Solution:
        base = list(self.population[index].representation)
        if len(base) < 2:
            return Solution(base, self.problem)
        if self._looks_like_permutation(base):
            i, j = self.rng.choice(len(base), size=2, replace=False)
            base[i], base[j] = base[j], base[i]
            return Solution(base, self.problem)
        candidate = base[:]
        pos = int(self.rng.integers(len(candidate)))
        value = candidate[pos]
        if self._is_binary(candidate):
            candidate[pos] = 1 - int(value)
        else:
            partner = list(self.population[partner_idx].representation)
            diff = partner[pos] - value if pos < len(partner) else self.rng.integers(-1, 2)
            step = int(np.sign(diff)) if diff != 0 else int(self.rng.choice([-1, 1]))
            candidate[pos] = value + step
        return Solution(candidate, self.problem)

    def _pick_partner(self, index: int) -> int:
        choice = index
        attempts = 0
        while choice == index and attempts < 5:
            choice = int(self.rng.integers(len(self.population)))
            attempts += 1
        if choice == index:
            choice = (index + 1) % len(self.population)
        return choice

    def _clip(self, vector: np.ndarray) -> np.ndarray:
        lower = self._lower_bounds
        upper = self._upper_bounds
        if lower is None and upper is None:
            return vector
        if lower is None:
            lower = np.full_like(vector, -np.inf)
        elif lower.size != vector.size:
            lower = np.full_like(vector, lower[0])
        if upper is None:
            upper = np.full_like(vector, np.inf)
        elif upper.size != vector.size:
            upper = np.full_like(vector, upper[0])
        return np.clip(vector, lower, upper)

    @staticmethod
    def _infer_dimension(solution: Solution) -> int:
        rep = solution.representation
        if isinstance(rep, (list, tuple)):
            return len(rep)
        try:
            return len(rep)
        except TypeError:
            return 1

    @staticmethod
    def _looks_like_permutation(values: Sequence) -> bool:
        if not values:
            return False
        ints = all(isinstance(v, (int, np.integer)) for v in values)
        if not ints:
            return False
        unique = len(set(values))
        return unique == len(values)

    @staticmethod
    def _is_binary(values: Sequence) -> bool:
        if not values:
            return False
        seen = {int(v) for v in values if isinstance(v, (int, np.integer, bool))}
        return seen.issubset({0, 1}) and bool(seen)

    @staticmethod
    def _as_array(bounds) -> Optional[np.ndarray]:
        if bounds is None:
            return None
        arr = np.asarray(bounds, dtype=float)
        if arr.ndim == 0:
            arr = np.array([float(arr)])
        return arr
