"""
Observation computation for the RL environment (problem-agnostic).
Implements the Orchestrator-V2 observation space.
"""

import math
from collections import deque
from typing import List, Optional

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from Core.problem import Solution, ProblemInterface


import logging

class ObservationComputer:
    """
    Compute the 8-dimensional Orchestrator-V2 observation vector from a solver state.
    This space is designed to be predictive and efficient, giving the RL agent a rich
    view of the search dynamics.

    Output layout (index -> name):
    0: budget_remaining
    1: normalized_best_fitness
    2: improvement_velocity
    3: stagnation_nonparametric
    4: population_concentration
    5: landscape_funnel_proxy
    6: landscape_deceptiveness_proxy
    7: active_phase
    """

    def __init__(
        self,
        problem_bounds: dict,
        *,
        velocity_ewma_alpha: float = 0.3,
        stagnation_window_size: int = 20,
        funnel_probe_size: int = 5,
        deception_probe_mutation_strength: float = 0.5,
        logger: logging.Logger,
    ):
        """
        Initializes the observation computer.
        Args:
            problem_bounds: Dict with fitness bounds ('lower_bound', 'upper_bound').
            velocity_ewma_alpha: Smoothing factor for improvement velocity.
            stagnation_window_size: Size of one window for the Mann-Whitney U test.
            funnel_probe_size: Number of neighbors to sample for the funnel proxy.
            deception_probe_mutation_strength: Strength of mutation for the deception probe.
        """
        self.logger = logger
        # Extract fitness bounds for normalization
        self.fitness_lower_bound, self.fitness_upper_bound = self._extract_bounds(problem_bounds)
        self.fitness_range = max(1e-9, self.fitness_upper_bound - self.fitness_lower_bound)

        # Parameters
        self.velocity_ewma_alpha = float(velocity_ewma_alpha)
        self.stagnation_window_size = max(4, int(stagnation_window_size))
        self.funnel_probe_size = max(2, int(funnel_probe_size))
        self.deception_probe_mutation_strength = float(deception_probe_mutation_strength)
        self.rng = np.random.default_rng()

        # State trackers
        self.step_index: int = 0
        self.prev_normalized_best_fitness: float = 1.0
        self.improvement_velocity: float = 0.0
        self.fitness_history: deque[float] = deque(maxlen=self.stagnation_window_size * 2)

        self.logger.debug("ObservationComputer initialized")
        self.logger.debug(f"  fitness_bounds: {self.fitness_lower_bound, self.fitness_upper_bound}")
        self.logger.debug(f"  velocity_ewma_alpha: {self.velocity_ewma_alpha}")
        self.logger.debug(f"  stagnation_window_size: {self.stagnation_window_size}")
        self.logger.debug(f"  funnel_probe_size: {self.funnel_probe_size}")
        self.logger.debug(f"  deception_probe_mutation_strength: {self.deception_probe_mutation_strength}")

    def reset(self) -> None:
        """Reset all internal state trackers for a new episode."""
        self.step_index = 0
        self.prev_normalized_best_fitness = 1.0
        self.improvement_velocity = 0.0
        self.fitness_history.clear()

    def compute(self, solver, phase: str, step_ratio: float) -> np.ndarray:
        """Compute the 8-element observation vector and log a detailed breakdown."""
        self.step_index += 1

        # Snapshot core solver state
        best_solution = solver.get_best()
        problem = solver.problem

        # --- Feature 1: budget_remaining ---
        # Normalized countdown of the remaining evaluation budget.
        budget_remaining = 1.0 - float(np.clip(step_ratio, 0.0, 1.0))

        # --- Feature 2: normalized_best_fitness ---
        # Progress gauge relative to known instance bounds.
        best_fitness = best_solution.fitness if best_solution and best_solution.fitness is not None else float("inf")
        normalized_best_fitness = (best_fitness - self.fitness_lower_bound) / self.fitness_range
        normalized_best_fitness = float(np.clip(normalized_best_fitness, 0.0, 1.0))

        # --- Feature 3: improvement_velocity ---
        # Smoothed rate of improvement in normalized best fitness.
        delta = self.prev_normalized_best_fitness - normalized_best_fitness
        self.improvement_velocity = (
            self.velocity_ewma_alpha * delta + (1.0 - self.velocity_ewma_alpha) * self.improvement_velocity
        )
        self.prev_normalized_best_fitness = normalized_best_fitness

        # --- Feature 4: stagnation_nonparametric ---
        # Statistical test for meaningful progress in recent history.
        self.fitness_history.append(best_fitness)
        stagnation = self._compute_stagnation()

        # --- Feature 5: population_concentration ---
        # Direct measure of population spread around its centroid.
        concentration = self._compute_population_concentration(solver)

        # --- Feature 6: landscape_funnel_proxy ---
        # Probes for smooth basin structure around the incumbent best solution.
        funnel = self._compute_landscape_funnel(best_solution, problem)

        # --- Feature 7: landscape_deceptiveness_proxy ---
        # Single long-jump evaluation to sense alternative basins.
        deceptiveness = self._compute_landscape_deceptiveness(best_solution, problem, normalized_best_fitness)

        # --- Feature 8: active_phase ---
        # Categorical encoding of the currently active solver phase.
        active_phase = 1.0 if phase == "exploitation" else 0.0

        # Assemble and return the final observation vector
        observation = np.array([
            budget_remaining,
            normalized_best_fitness,
            float(np.clip(self.improvement_velocity, -1.0, 1.0)),
            stagnation,
            concentration,
            funnel,
            deceptiveness,
            active_phase,
        ], dtype=np.float32)
        # Optional detailed breakdown (debug-level only to keep file logs high-level)
        try:
            self.logger.debug("Observation calculation:")
            self.logger.debug(
                f"  - step: {self.step_index}, phase: {phase}, step_ratio: {float(step_ratio):.4f}"
            )
            self.logger.debug(f"  - budget_remaining: {float(budget_remaining):.4f}")
            self.logger.debug(f"  - normalized_best_fitness: {float(normalized_best_fitness):.4f}")
            self.logger.debug(
                f"  - improvement_velocity: {float(self.improvement_velocity):.4f} (delta: {float(delta):.4f})"
            )
            self.logger.debug(f"  - stagnation_nonparametric: {float(stagnation):.4f}")
            self.logger.debug(f"  - population_concentration: {float(concentration):.4f}")
            self.logger.debug(f"  - landscape_funnel_proxy: {float(funnel):.4f}")
            self.logger.debug(f"  - landscape_deceptiveness_proxy: {float(deceptiveness):.4f}")
            self.logger.debug(f"  - active_phase: {'exploitation' if active_phase == 1.0 else 'exploration'}")
        except Exception:
            pass
        self.logger.debug(f"Observation computed at step {self.step_index}: {observation}")
        return observation

    def _compute_stagnation(self) -> float:
        """Computes stagnation via Mann-Whitney U test; returns 1.0 - p-value."""
        history = list(self.fitness_history)
        history_len = len(history)

        # Require a minimal amount of history before emitting any stagnation signal.
        if history_len < max(4, self.stagnation_window_size // 2):
            return 0.0

        window_size = min(self.stagnation_window_size, history_len // 2)
        if window_size < 2:
            return 0.0

        window1 = history[:window_size]
        window2 = history[-window_size:]

        # If windows are identical, U-test is meaningless, implies stagnation.
        if np.allclose(window1, window2):
            return 1.0
        
        try:
            _, p_value = mannwhitneyu(window1, window2, alternative='two-sided')
            return 1.0 - float(p_value)
        except ValueError:
            # Occurs if all values in a window are the same. Treat as stagnation.
            return 1.0

    def _compute_population_concentration(self, solver) -> float:
        """Computes population concentration, optimized for vectorized solvers."""
        # Check if the solver is vectorized
        if hasattr(solver, '_population_matrix') and solver._population_matrix is not None:
            reps_arr = solver._population_matrix
        else:
            # Fallback for non-vectorized solvers
            population = solver.get_population()
            if not population or len(population) < 2:
                return 1.0
            reps = [sol.representation for sol in population if sol and sol.representation is not None]
            if not reps:
                return 1.0
            reps_arr = np.array(reps, dtype=float)

        if reps_arr.shape[0] < 2:
            return 1.0

        try:
            if reps_arr.ndim == 1: reps_arr = reps_arr.reshape(-1, 1)
            
            min_vals, max_vals = np.min(reps_arr, axis=0), np.max(reps_arr, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals < 1e-9] = 1.0
            normalized_reps = (reps_arr - min_vals) / range_vals

            centroid = np.mean(normalized_reps, axis=0)
            distances = np.linalg.norm(normalized_reps - centroid, axis=1)
            mean_dist = np.mean(distances)
            
            num_dims = reps_arr.shape[1]
            if num_dims <= 0: return 1.0
            
            scaled_dist = mean_dist / math.sqrt(num_dims)
            return 1.0 - float(np.clip(scaled_dist, 0.0, 1.0))
        except (ValueError, TypeError):
            return 0.5

    def _compute_landscape_funnel(self, best_solution: Optional[Solution], problem: ProblemInterface) -> float:
        """Computes funnel proxy via Spearman correlation, using batch evaluation."""
        if not best_solution or not hasattr(problem, 'evaluate') or not hasattr(problem, 'nkl_problem'):
            return 0.0

        # --- Vectorized Neighbor Generation ---
        best_rep = np.asarray(best_solution.representation)
        # Create N copies of the best solution to mutate
        candidate_reps = np.tile(best_rep, (self.funnel_probe_size, 1))
        
        # Create N random mutation masks (strength 0.05)
        mutation_strength = 0.05
        mutation_mask = self.rng.random(candidate_reps.shape) < mutation_strength
        
        # Ensure each probe is at least slightly different
        no_flips_mask = ~np.any(mutation_mask, axis=1)
        if np.any(no_flips_mask):
            random_indices = self.rng.integers(0, candidate_reps.shape[1], size=np.sum(no_flips_mask))
            mutation_mask[no_flips_mask, random_indices] = True

        neighbor_reps = np.where(mutation_mask, 1 - candidate_reps, candidate_reps)

        # --- Batch Evaluation ---
        try:
            # Ensure binary and correct dimensionality in case the instance was regenerated
            neighbor_reps = np.where(neighbor_reps > 0, 1, 0).astype(int)
            n = int(getattr(problem.nkl_problem, 'n', neighbor_reps.shape[1]))
            if neighbor_reps.shape[1] != n:
                if neighbor_reps.shape[1] > n:
                    neighbor_reps = neighbor_reps[:, :n]
                else:
                    pad_cols = n - neighbor_reps.shape[1]
                    pad = self.rng.integers(0, 2, size=(neighbor_reps.shape[0], pad_cols), dtype=int)
                    neighbor_reps = np.concatenate([neighbor_reps, pad], axis=1)

            # Align best_rep as well for distance computation
            best_rep_bin = np.where(best_rep > 0, 1, 0).astype(int)
            if best_rep_bin.shape[0] != n:
                if best_rep_bin.shape[0] > n:
                    best_rep_bin = best_rep_bin[:n]
                else:
                    pad_cols = n - best_rep_bin.shape[0]
                    pad = self.rng.integers(0, 2, size=(pad_cols,), dtype=int)
                    best_rep_bin = np.concatenate([best_rep_bin, pad], axis=0)

            neighbor_fitnesses = problem.nkl_problem.evaluate_batch(neighbor_reps)
        except Exception:
            return 0.0 # Fallback if batch evaluation fails

        # --- Vectorized Distance Calculation ---
        distances = np.linalg.norm(neighbor_reps - best_rep_bin, axis=1)

        if len(distances) < 2:
            return 0.0

        if np.allclose(distances, distances[0]) or np.allclose(neighbor_fitnesses, neighbor_fitnesses[0]):
            return 0.0
        
        try:
            corr, _ = spearmanr(distances, neighbor_fitnesses)
            return float(np.clip(corr, -1.0, 1.0)) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def _compute_landscape_deceptiveness(self, best_solution: Optional[Solution], problem: ProblemInterface, norm_best_fit: float) -> float:
        """Computes deceptiveness proxy with a long-jump mutation."""
        if not best_solution or not hasattr(problem, 'evaluate'):
            return 0.0

        # Create a heavily mutated solution to probe a distant region
        far_solution = self._mutate_solution(best_solution, problem, strength=self.deception_probe_mutation_strength)
        far_solution.evaluate()

        if far_solution.fitness is None or not math.isfinite(far_solution.fitness):
            return 0.0

        # Normalize the fitness of the distant probe
        norm_far_fit = (far_solution.fitness - self.fitness_lower_bound) / self.fitness_range
        norm_far_fit = float(np.clip(norm_far_fit, 0.0, 1.0))

        # Deceptiveness is high if the far solution is better than the current best
        return float(np.clip(norm_best_fit - norm_far_fit, -1.0, 1.0))

    def _mutate_solution(self, solution: Solution, problem: ProblemInterface, strength: float) -> Solution:
        """Problem-aware mutation. Adapts to representation type using problem_info."""
        rep = solution.representation
        info = problem.get_problem_info()
        problem_type = info.get("problem_type")
        new_rep = None

        try:
            if problem_type == "binary":
                # Bit-flip mutation for binary vectors (Knapsack, Max-Cut)
                rep_arr = np.array(rep)
                flip_prob = strength
                mutation_mask = self.rng.random(size=rep_arr.shape) < flip_prob
                # Ensure at least one flip for non-zero strength
                if strength > 0 and not np.any(mutation_mask):
                    idx_to_flip = self.rng.integers(0, len(rep_arr))
                    mutation_mask[idx_to_flip] = True
                new_rep = np.where(mutation_mask, 1 - rep_arr, rep_arr).tolist()
                # Repair if the problem requires it (e.g., Knapsack capacity)
                if hasattr(problem, 'repair_mask'):
                    new_rep = problem.repair_mask(new_rep).tolist()

            elif problem_type == "permutation":
                # Swap or 2-opt style mutation for permutations (TSP)
                new_rep = list(rep)
                if len(new_rep) < 2: return solution.copy()
                num_mutations = max(1, int(strength * len(new_rep)))
                for _ in range(num_mutations):
                    i, j = self.rng.choice(range(len(new_rep)), 2, replace=False)
                    if self.rng.random() < 0.5:  # 2-opt like inversion
                        if i > j: i, j = j, i
                        sub = new_rep[i:j+1]
                        sub.reverse()
                        new_rep = new_rep[:i] + sub + new_rep[j+1:]
                    else:  # Swap
                        new_rep[i], new_rep[j] = new_rep[j], new_rep[i]
            
            elif isinstance(rep, (np.ndarray, list)):
                # Fallback for continuous vector problems: add Gaussian noise
                rep_arr = np.array(rep, dtype=float)
                noise = self.rng.normal(0, strength, size=rep_arr.shape)
                new_rep_arr = rep_arr + noise
                # Clip to bounds if they exist
                if 'lower_bounds' in info and 'upper_bounds' in info:
                    new_rep_arr = np.clip(new_rep_arr, info['lower_bounds'], info['upper_bounds'])
                new_rep = new_rep_arr.tolist() if isinstance(rep, list) else new_rep_arr

        except Exception:
            pass  # Fallback on any error

        if new_rep is not None:
            return Solution(new_rep, problem)
        else:
            # Ultimate fallback for unsupported types or errors
            return problem.get_initial_solution()

    def _solution_distance(self, sol1: Solution, sol2: Solution) -> float:
        """
        Computes Euclidean distance between two solutions' representations.
        Note: This is a generic proxy. For permutation-based problems like TSP,
        Euclidean distance is not a true metric of tour similarity but can still
        provide a useful signal for concentration.
        """
        if not sol1 or not sol2 or sol1.representation is None or sol2.representation is None:
            return 0.0
        try:
            rep1 = np.asarray(sol1.representation, dtype=float)
            rep2 = np.asarray(sol2.representation, dtype=float)
            return float(np.linalg.norm(rep1 - rep2))
        except (ValueError, TypeError):
            return 0.0 # Fallback for non-numeric representations

    @staticmethod
    def _extract_bounds(meta: dict) -> tuple[float, float]:
        if not isinstance(meta, dict):
            return 0.0, 1.0
        # Direct
        if "lower_bound" in meta and "upper_bound" in meta:
            try:
                lb = float(meta["lower_bound"])
                ub = float(meta["upper_bound"])
                if math.isfinite(lb) and math.isfinite(ub) and ub > lb:
                    return lb, ub
            except Exception:
                pass
        # fitness_* variants
        for lo_key, hi_key in (("fitness_lower_bound", "fitness_upper_bound"), ("fitness_min", "fitness_max")):
            if lo_key in meta and hi_key in meta:
                try:
                    lb = float(meta[lo_key])
                    ub = float(meta[hi_key])
                    if math.isfinite(lb) and math.isfinite(ub) and ub > lb:
                        return lb, ub
                except Exception:
                    continue
        # Fallback
        return 0.0, 1.0
