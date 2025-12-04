"""
Observation computation for the RL environment (problem-agnostic).
Implements the minimal 6-dimensional observation space.

Observation layout (index -> name):
    0: budget_consumed        - Budget used so far (0→1)
    1: fitness_norm           - Normalized fitness (0=optimal, 1=worst)
    2: improvement_velocity   - EWMA-smoothed improvement signal (0→1)
    3: stagnation             - Search stagnation level (0→1, gradual)
    4: diversity              - Population diversity (0→1)
    5: phase                  - Current phase (0.0=explore, 0.5=exploit, 1.0=terminate)
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
    
    This observation space is designed to work with the Effectiveness-First 
    Reward (EFR) function. Key design choices:
    
    1. budget_consumed (not remaining): Aligns with reward function that uses
       budget_consumed for pressure calculations.
    
    2. improvement_velocity: EWMA-smoothed improvement signal that captures
       both whether improvements happened and their relative magnitude.
       More informative than binary 0/1.
    
    3. Gradual stagnation: Grows slowly to allow meaningful exploration before
       the agent sees "high stagnation". Prevents premature phase switches.
    
    4. 3-phase encoding: Supports exploration (0.0), exploitation (0.5), and
       termination (1.0) phases explicitly.
    """

    feature_names = [
        "budget_consumed",
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
        stagnation_window: int = 10,
        ewma_alpha: float = 0.3,
        logger: logging.Logger,
    ):
        self.logger = logger
        self.fitness_lower_bound, self.fitness_upper_bound = self._extract_bounds(problem_meta)
        self.fitness_range = max(1e-9, self.fitness_upper_bound - self.fitness_lower_bound)
        
        self.stagnation_window = max(2, int(stagnation_window))
        self.ewma_alpha = ewma_alpha  # EWMA smoothing factor (higher = more weight to recent)
        
        # Internal state
        self.fitness_history: deque[float] = deque(maxlen=self.stagnation_window)
        self._last_best_fitness: float = float('inf')
        self._improvement_velocity_ewma: float = 0.0  # EWMA of improvement velocity
        
        # Diversity caching for performance
        self._cached_diversity: Optional[float] = None
        self._last_population_hash: Optional[int] = None
        
        self.logger.debug("ObservationComputer initialized with EWMA improvement velocity.")

    def reset(self) -> None:
        """Reset all internal state trackers for a new episode."""
        self.fitness_history.clear()
        self._last_best_fitness = float('inf')
        self._improvement_velocity_ewma = 0.0
        self._cached_diversity = None
        self._last_population_hash = None

    def compute(self, state: ObservationState) -> np.ndarray:
        """
        Compute the 6-element observation vector.
        
        Args:
            state: ObservationState with solver, phase, step_ratio, etc.
            
        Returns:
            np.ndarray of shape (6,) with values in [0, 1]
        """
        solver = state.solver
        phase = state.phase
        step_ratio = state.step_ratio
        best_solution = state.best_solution or solver.get_best()
        
        # === 0. Budget Consumed (0 at start, 1 at end) ===
        budget_consumed = float(np.clip(step_ratio, 0.0, 1.0))
        
        # === 1. Normalized Fitness (0=optimal, 1=worst) ===
        best_fitness = best_solution.fitness if best_solution and best_solution.fitness is not None else float('inf')
        
        if self.fitness_range <= 1e-9 or not np.isfinite(best_fitness) or best_fitness == float('inf'):
            fitness_norm = 1.0  # Worst quality if no valid fitness
        else:
            fitness_norm = (best_fitness - self.fitness_lower_bound) / self.fitness_range
            fitness_norm = float(np.clip(fitness_norm, 0.0, 1.0))
            if not np.isfinite(fitness_norm):
                fitness_norm = 1.0
        
        # === 2. Improvement Velocity (EWMA smoothed) ===
        improvement_this_step = 0.0
        if np.isfinite(best_fitness) and best_fitness < self._last_best_fitness - 1e-9:
            # Calculate relative improvement magnitude (normalized by fitness range)
            improvement_magnitude = (self._last_best_fitness - best_fitness) / self.fitness_range
            improvement_this_step = min(1.0, improvement_magnitude * 10.0)  # Scale and cap at 1.0
            self._last_best_fitness = best_fitness
        elif np.isfinite(best_fitness) and self._last_best_fitness == float('inf'):
            # First valid fitness - moderate improvement signal
            self._last_best_fitness = best_fitness
            improvement_this_step = 0.5  # Moderate signal for first valid fitness
        
        # Update EWMA: velocity = alpha * current + (1-alpha) * previous
        self._improvement_velocity_ewma = (
            self.ewma_alpha * improvement_this_step + 
            (1.0 - self.ewma_alpha) * self._improvement_velocity_ewma
        )
        improvement_velocity = float(np.clip(self._improvement_velocity_ewma, 0.0, 1.0))
        
        # Track fitness history (only finite values)
        if np.isfinite(best_fitness):
            self.fitness_history.append(best_fitness)
        
        # === 3. Stagnation (gradual measure of search progress) ===
        stagnation = self._compute_stagnation()
        
        # === 4. Population Diversity ===
        population = state.population if state.population is not None else solver.get_population()
        diversity = self._compute_population_diversity(population)
        
        # === 5. Phase Encoding (3-phase: 0.0/0.5/1.0) ===
        phase_encoding = {
            "exploration": 0.0,
            "exploitation": 0.5,
            "termination": 1.0
        }
        active_phase = phase_encoding.get(phase, 0.0)
        
        # Build observation vector
        observation = np.array([
            budget_consumed,       # 0: budget used (0→1)
            fitness_norm,          # 1: quality inverse (0=best, 1=worst)
            improvement_velocity,  # 2: EWMA of improvement velocity (0→1, smoothed)
            stagnation,            # 3: search stuck? (0→1)
            diversity,             # 4: population spread (0→1)
            active_phase           # 5: phase (0.0/0.5/1.0)
        ], dtype=np.float32)
        
        # Replace any NaN/inf with safe defaults
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=0.0)
        
        self.logger.debug(f"Observation: {observation}")
        return observation

    def _compute_stagnation(self) -> float:
        """
        Compute stagnation as a gradual measure of search progress.
        
        Key design:
        1. Requires minimum history (15 steps) before reporting high stagnation
        2. Uses improvement rate over the window
        3. Only reaches 0.7+ after sustained lack of improvement
        4. Capped at 0.85 until very long stagnation
        
        This prevents premature phase switching during early exploration.
        """
        min_history_for_stagnation = 15  # Need more samples before high stagnation
        warmup_period = 8  # Steps before any significant stagnation
        
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Filter out any non-finite values
        finite_history = [f for f in self.fitness_history if np.isfinite(f)]
        if len(finite_history) < 2:
            return 0.0
        
        history_len = len(finite_history)
        
        # During warmup, keep stagnation very low
        if history_len < warmup_period:
            # Linear ramp from 0 to 0.3 during warmup
            return 0.3 * (history_len / warmup_period)
        
        # Count improvements: how many times did fitness decrease?
        improvements = 0
        for i in range(1, len(finite_history)):
            if finite_history[i] < finite_history[i-1] - 1e-9:
                improvements += 1
        
        # Improvement rate: what fraction of steps had improvement?
        improvement_rate = improvements / (len(finite_history) - 1)
        
        # Also check overall improvement from start to now
        overall_improvement = (finite_history[0] - finite_history[-1]) / (abs(finite_history[0]) + 1e-9)
        overall_improvement = max(0, overall_improvement)  # Only positive improvement counts
        
        # Base stagnation: low improvement rate = high stagnation
        # If we're improving 30%+ of steps, stagnation is 0
        # If we're improving 0% of steps, stagnation approaches 1.0
        rate_stagnation = 1.0 - np.clip(improvement_rate * 3.33, 0.0, 1.0)
        
        # Overall improvement bonus: if we're still improving overall, reduce stagnation
        overall_bonus = np.clip(overall_improvement * 2.0, 0.0, 0.3)
        
        base_stagnation = max(0.0, rate_stagnation - overall_bonus)
        
        # Confidence scaling: ramp from 0.4 to 1.0 based on history
        if history_len < min_history_for_stagnation:
            confidence = 0.4 + 0.6 * (history_len - warmup_period) / (min_history_for_stagnation - warmup_period)
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            confidence = 1.0
        
        stagnation = base_stagnation * confidence
        
        # Ensure finite result and cap at 0.95 until very long stagnation
        if not np.isfinite(stagnation):
            return 0.0
        
        # Only allow stagnation > 0.9 if truly stuck for long time
        if history_len < min_history_for_stagnation * 2:
            stagnation = min(stagnation, 0.85)
        
        return float(stagnation)

    def _compute_population_diversity(self, population: Optional[List[Solution]]) -> float:
        """
        Computes population diversity as the mean distance from centroid.
        
        Uses caching to avoid redundant calculations when population hasn't changed.
        """
        if not population or len(population) < 2:
            return 0.0
        
        # Filter out solutions with None representations
        valid_population = [sol for sol in population if sol and sol.representation is not None]
        if len(valid_population) < 2:
            return 0.0
        
        # Check if we can use cached diversity
        try:
            current_hash = hash(tuple(
                sol.representation.tobytes() if hasattr(sol.representation, 'tobytes')
                else str(sol.representation).encode()
                for sol in valid_population
            ))
        except (AttributeError, TypeError):
            current_hash = hash(tuple(str(sol.representation) for sol in valid_population))
        
        if self._last_population_hash == current_hash and self._cached_diversity is not None:
            return self._cached_diversity
        
        # Calculate diversity
        try:
            reps = np.array([np.asarray(sol.representation) for sol in valid_population])
            if reps.shape[0] < 2:
                return 0.0
            
            # Handle single-dimensional arrays
            if reps.ndim == 1:
                reps = reps.reshape(1, -1)
            
            # Normalize each dimension to [0, 1]
            reps_min = reps.min(axis=0)
            reps_max = reps.max(axis=0)
            reps_range = np.maximum(reps_max - reps_min, 1e-9)
            normalized_reps = (reps - reps_min) / reps_range
            
            # Calculate mean distance from centroid
            centroid = np.mean(normalized_reps, axis=0)
            distances = np.linalg.norm(normalized_reps - centroid[None, :], axis=1)
            mean_dist = np.mean(distances)
            
            # Scale to [0, 1]
            # Max possible mean distance in a hypercube is sqrt(num_dims) / 2
            max_possible_dist = math.sqrt(reps.shape[1]) / 2
            diversity = float(np.clip(mean_dist / (max_possible_dist + 1e-9), 0.0, 1.0))
            
            if not np.isfinite(diversity):
                diversity = 0.0
            
            # Cache result
            self._cached_diversity = diversity
            self._last_population_hash = current_hash
            
            return diversity
            
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0

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
