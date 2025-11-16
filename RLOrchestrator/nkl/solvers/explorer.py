"""
Fully vectorized exploration solver for NKL.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


class NKLRandomExplorer(SearchAlgorithm):
    """Maintains a diverse population via vectorized random bit flips."""
    phase = 'exploration'

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 64,
        *,
        flip_probability: float = 0.15,
        elite_fraction: float = 0.25,
        seed: Optional[int] = None,
    ):
        super().__init__(problem, population_size)
        self.flip_probability = float(np.clip(flip_probability, 0.01, 0.9))
        self.elite_fraction = float(np.clip(elite_fraction, 0.05, 0.8))
        self.rng = np.random.default_rng(seed)
        
        # Internally, population is a numpy array for performance
        self._population_matrix: Optional[np.ndarray] = None
        self._fitness_values: Optional[np.ndarray] = None

    def initialize(self):
        n = self.problem.get_problem_info()["dimension"]
        self._population_matrix = self.rng.integers(0, 2, size=(self.population_size, n), dtype=np.int8)
        self._fitness_values = self.problem.nkl_problem.evaluate_batch(self._population_matrix)
        self._update_best_solution_from_matrix()

    def step(self):
        if self._population_matrix is None or self._fitness_values is None:
            self.initialize()

        # --- 1. Select Elites ---
        elite_count = max(1, int(self.elite_fraction * self.population_size))
        elite_indices = np.argsort(self._fitness_values)[:elite_count]
        elites = self._population_matrix[elite_indices]
        elite_fitnesses = self._fitness_values[elite_indices]

        # --- 2. Generate Offspring (Vectorized) ---
        # Create mutations for the entire population
        mutation_mask = self.rng.random(self._population_matrix.shape) < self.flip_probability
        
        # Ensure at least one flip per offspring to prevent stagnation
        no_flips_mask = ~np.any(mutation_mask, axis=1)
        if np.any(no_flips_mask):
            random_indices = self.rng.integers(0, self._population_matrix.shape[1], size=np.sum(no_flips_mask))
            mutation_mask[no_flips_mask, random_indices] = True

        # Apply mutations to create children
        offspring_matrix = np.where(mutation_mask, 1 - self._population_matrix, self._population_matrix)

        # --- 3. Evaluate Offspring (Batch) ---
        offspring_fitnesses = self.problem.nkl_problem.evaluate_batch(offspring_matrix)

        # --- 4. Combine and Select New Population ---
        combined_population = np.vstack([elites, offspring_matrix])
        combined_fitnesses = np.concatenate([elite_fitnesses, offspring_fitnesses])

        # Select the best for the next generation
        selection_indices = np.argsort(combined_fitnesses)[:self.population_size]
        self._population_matrix = combined_population[selection_indices]
        self._fitness_values = combined_fitnesses[selection_indices]

        # --- 5. Update Best Solution ---
        self._update_best_solution_from_matrix()
        self.iteration += 1

    def _update_best_solution_from_matrix(self):
        best_idx = np.argmin(self._fitness_values)
        best_fitness = self._fitness_values[best_idx]
        if self.best_solution is None or best_fitness < self.best_solution.fitness:
            best_representation = self._population_matrix[best_idx].tolist()
            self.best_solution = Solution(best_representation, self.problem)
            self.best_solution.fitness = best_fitness

    def get_population(self) -> list[Solution]:
        # This is a compatibility method for the observation computer.
        # It's inefficient and should be used sparingly.
        solutions = []
        if self._population_matrix is not None and self._fitness_values is not None:
            for rep, fit in zip(self._population_matrix, self._fitness_values):
                sol = Solution(rep.tolist(), self.problem)
                sol.fitness = fit
                solutions.append(sol)
        return solutions