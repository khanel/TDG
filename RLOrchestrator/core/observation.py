"""
Observation computation for the RL environment (problem-agnostic).
Implements the minimal 6-dimensional observation space.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import logging

from Core.problem import Solution
from Core.search_algorithm import SearchAlgorithm


@dataclass
class ObservationState:
    """Structured snapshot of the orchestrator state for observation computation."""
    solver: SearchAlgorithm
    phase: str
    step_ratio: float
    best_solution: Optional[Solution] = None
    population: Optional[List[Solution]] = None

class ObservationComputer:
    """
    Computes the 6-dimensional observation vector from a solver state.
    This space is designed to be lean and informative.

    Output layout (index -> name):
    0: budget_remaining
    1: normalized_best_fitness
    2: improvement_velocity
    3. stagnation
    4. population_diversity
    5: active_phase
    """

    feature_names = [
        "budget_remaining",
        "normalized_best_fitness",
        "improvement_velocity",
        "stagnation",
        "population_diversity",
        "active_phase",
    ]

    def __init__(
        self,
        problem_meta: dict,
        *,
        velocity_ewma_alpha: float = 0.3,
        stagnation_window: int = 10,
        logger: logging.Logger,
    ):
        self.logger = logger
        self.fitness_lower_bound, self.fitness_upper_bound = self._extract_bounds(problem_meta)
        self.fitness_range = max(1e-9, self.fitness_upper_bound - self.fitness_lower_bound)

        self.velocity_ewma_alpha = float(velocity_ewma_alpha)
        self.stagnation_window = max(2, int(stagnation_window))
        self.rng = np.random.default_rng()

        self.step_index: int = 0
        self.prev_normalized_best_fitness: float = 1.0
        self.improvement_velocity: float = 0.0
        self.fitness_history: deque[float] = deque(maxlen=self.stagnation_window)
        self.prev_diversity: float = 0.0
        self.diversity_collapse_rate: float = 0.0
        
        self.logger.debug("ObservationComputer initialized.")

    def reset(self) -> None:
        """Reset all internal state trackers for a new episode."""
        self.step_index = 0
        self.prev_normalized_best_fitness = 1.0
        self.improvement_velocity = 0.0
        self.fitness_history.clear()
        self.prev_diversity = 0.0
        self.diversity_collapse_rate = 0.0

    def compute(self, state: ObservationState) -> np.ndarray:
        """Compute the 6-element observation vector."""
        solver = state.solver
        phase = state.phase
        step_ratio = state.step_ratio
        self.step_index += 1
        best_solution = state.best_solution or solver.get_best()

        # 1. budget_remaining
        budget_remaining = 1.0 - float(np.clip(step_ratio, 0.0, 1.0))

        # 2. normalized_best_fitness
        best_fitness = best_solution.fitness if best_solution else float("inf")
        normalized_best_fitness = (best_fitness - self.fitness_lower_bound) / self.fitness_range
        normalized_best_fitness = float(np.clip(normalized_best_fitness, 0.0, 1.0))
        self.fitness_history.append(best_fitness)

        # 3. improvement_velocity
        delta = self.prev_normalized_best_fitness - normalized_best_fitness
        self.improvement_velocity = (
            self.velocity_ewma_alpha * delta + (1.0 - self.velocity_ewma_alpha) * self.improvement_velocity
        )
        self.prev_normalized_best_fitness = normalized_best_fitness

        # 4. stagnation
        stagnation = self._compute_stagnation()

        # 5. population_diversity
        population = state.population if state.population is not None else solver.get_population()
        diversity = self._compute_population_diversity(population)
        div_delta = self.prev_diversity - diversity
        self.diversity_collapse_rate = (
            self.velocity_ewma_alpha * div_delta
            + (1.0 - self.velocity_ewma_alpha) * self.diversity_collapse_rate
        )
        self.prev_diversity = diversity

        # 6. active_phase
        active_phase = 1.0 if phase == "exploitation" else 0.0

        features = [
            budget_remaining,
            normalized_best_fitness,
            float(np.clip(self.improvement_velocity, -1.0, 1.0)),
            stagnation,
            diversity,
            active_phase,
        ]

        observation = np.array(features, dtype=np.float32)
        
        self.logger.debug(f"Observation: {observation}")
        return observation

    def _compute_stagnation(self) -> float:
        if len(self.fitness_history) < self.stagnation_window:
            return 0.0
        
        first_val = self.fitness_history[0]
        last_val = self.fitness_history[-1]

        if first_val == last_val:
            return 1.0 # Stagnated
        
        return 0.0

    def _compute_population_diversity(self, population: Optional[List[Solution]]) -> float:
        """Computes population diversity as the mean pairwise distance."""
        if not population or len(population) < 2:
            return 0.0
        
        reps = [sol.representation for sol in population if sol and sol.representation is not None]
        if not reps or len(reps) < 2:
            return 0.0
            
        reps_arr = np.array(reps, dtype=float)
        
        # Normalize each dimension to [0, 1]
        min_vals, max_vals = np.min(reps_arr, axis=0), np.max(reps_arr, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-9] = 1.0
        normalized_reps = (reps_arr - min_vals) / range_vals
        
        # Calculate mean pairwise distance
        centroid = np.mean(normalized_reps, axis=0)
        distances = np.linalg.norm(normalized_reps - centroid, axis=1)
        mean_dist = np.mean(distances)
        
        # Scale to [0, 1]
        num_dims = reps_arr.shape[1]
        if num_dims <= 0: return 0.0
        
        # The maximum possible mean distance in a hypercube is sqrt(num_dims) / 2
        max_possible_dist = math.sqrt(num_dims) / 2
        return float(np.clip(mean_dist / max_possible_dist, 0.0, 1.0))

    @staticmethod
    def _extract_bounds(meta: dict) -> tuple[float, float]:
        """Extracts fitness bounds from problem metadata."""
        if not isinstance(meta, dict):
            return 0.0, 1.0
        
        keys_to_try = [
            ("lower_bound", "upper_bound"),
            ("fitness_lower_bound", "fitness_upper_bound"),
            ("fitness_min", "fitness_max"),
        ]
        
        for lo_key, hi_key in keys_to_try:
            if lo_key in meta and hi_key in meta:
                try:
                    lb = float(meta[lo_key])
                    ub = float(meta[hi_key])
                    if math.isfinite(lb) and math.isfinite(ub) and ub > lb:
                        return lb, ub
                except (ValueError, TypeError):
                    continue
                    
        return 0.0, 1.0
