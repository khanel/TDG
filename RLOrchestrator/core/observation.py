"""
Observation computation for the RL environment (problem-agnostic).

What this provides (7 features in [0,1]):
- phase_is_exploitation: 0 for exploration, 1 for exploitation.
- normalized_best_fitness: best fitness normalized to [0,1] using provided bounds (lower is better).
- frontier_improvement_flag: 1 if the elite frontier (top 10% cutoff) improved significantly this step.
- frontier_success_rate: fraction of recent steps (window W) with frontier_improvement_flag = 1.
- elite_turnover_entropy: normalized Shannon entropy of elite IDs over recent snapshots.
- frontier_stagnation_ratio: steps since last frontier improvement divided by W (capped at 1).
- budget_used_ratio: current step / max episode steps.

Design goals:
- Conservative cost: no per-step full sorts. Periodic O(N log K) refresh + O(log K) incremental.
- Shared work: elite frontier powers imp1_top, sr_top, and elite_entropy.
"""

import hashlib
import heapq
import math
from collections import Counter, deque
from typing import List, Iterable, Optional

import numpy as np
from Core.problem import Solution


class ObservationComputer:
    """Compute observation vectors from a solver state with shared, low-cost metrics.

    Output layout (index -> name):
    0: phase_is_exploitation
    1: normalized_best_fitness
    2: frontier_improvement_flag
    3: frontier_success_rate
    4: elite_turnover_entropy
    5: frontier_stagnation_ratio
    6: budget_used_ratio
    """

    def __init__(
        self,
        problem_bounds: dict,
        *,
        success_window_size: int = 32,
        improvement_ewma_alpha: float = 0.1,
        improvement_gate_coeff: float = 0.5,
        elite_top_fraction: float = 0.10,
        elite_refresh_period: int = 10,
        elite_entropy_window_size: int = 32,
    ):
        # Accept either explicit bounds or problem info dicts; fall back to [0,1]
        self.fitness_lower_bound, self.fitness_upper_bound = self._extract_bounds(problem_bounds)
        self.fitness_range = max(1e-9, self.fitness_upper_bound - self.fitness_lower_bound)

        # Parameters
        self.success_window_size = max(2, int(success_window_size))  # W: window for success rate and stagnation
        self.improvement_ewma_alpha = float(improvement_ewma_alpha)  # α: EWMA smoothing for |Δcutoff|
        self.improvement_gate_coeff = float(improvement_gate_coeff)  # c: gating coefficient for significance
        self.elite_top_fraction = float(max(0.01, min(0.5, elite_top_fraction)))
        self.elite_refresh_period = max(1, int(elite_refresh_period))
        self.elite_entropy_window_size = max(1, int(elite_entropy_window_size))

        # Elite frontier trackers (top 10%). We store (-fitness, id) to make a max-heap by fitness.
        self.elite_heap: List[tuple] = []                 # max-heap over fitness via negative values
        self.elite_id_set: set[str] = set()               # set of current elite IDs
        self.elite_size: int = 0                          # K = ceil(fraction * N)
        self.elite_worst_fitness: float = math.inf        # current cutoff (worst elite fitness)
        self.prev_elite_worst_fitness: float = math.inf   # previous cutoff
        self.ewma_cutoff_change: float = 0.0              # EWMA(|Δcutoff|)
        self.improvement_flag_window: deque[int] = deque(maxlen=self.success_window_size)
        self.improvement_flag_sum: int = 0
        self.steps_since_elite_improvement: int = 0
        self.elites_initialized: bool = False
        self.step_index: int = 0

        # Entropy over elite IDs (snapshots taken only on refresh steps)
        self.elite_snapshot_window: deque[List[str]] = deque(maxlen=self.elite_entropy_window_size)
        self.elite_id_counts: Counter[str] = Counter()
        self.cached_entropy_value: float = 0.0

    def reset(self) -> None:
        # Elite state
        self.elite_heap.clear()
        self.elite_id_set.clear()
        self.elite_size = 0
        self.elite_worst_fitness = math.inf
        self.prev_elite_worst_fitness = math.inf
        self.ewma_cutoff_change = 0.0
        self.improvement_flag_window.clear()
        self.improvement_flag_sum = 0
        self.steps_since_elite_improvement = 0
        self.elites_initialized = False
        self.step_index = 0
        # Entropy state
        self.elite_snapshot_window.clear()
        self.elite_id_counts.clear()
        self.cached_entropy_value = 0.0

    def compute(self, solver, phase: str, step_ratio: float) -> np.ndarray:
        """Compute 7-element observation vector in the documented order."""
        self.step_index += 1

        # Snapshot inputs once to share across metrics
        population = solver.get_population()
        best_solution = solver.get_best()
        best_fitness = best_solution.fitness if best_solution and best_solution.fitness is not None else float("inf")

        # Normalize best (kept for compatibility; higher value here means worse if minimization)
        normalized_best_fitness = (best_fitness - self.fitness_lower_bound) / self.fitness_range

        # Update top-10% elites (refresh periodically; cheap incremental between refreshes)
        frontier_improvement_flag = self._update_elite_frontier(population, best_solution)
        frontier_success_rate = (self.improvement_flag_sum / len(self.improvement_flag_window)) if self.improvement_flag_window else 0.0
        elite_turnover_entropy = self.cached_entropy_value  # updated on refresh steps
        frontier_stagnation_ratio = (
            min(1.0, self.steps_since_elite_improvement / float(self.success_window_size))
            if self.success_window_size > 0 else 0.0
        )

        budget_used_ratio = float(np.clip(step_ratio, 0.0, 1.0))
        phase_is_exploitation = 1.0 if phase == "exploitation" else 0.0

        return np.array([
            phase_is_exploitation,
            float(np.clip(normalized_best_fitness, 0.0, 1.0)),
            float(frontier_improvement_flag),
            float(frontier_success_rate),
            float(np.clip(elite_turnover_entropy, 0.0, 1.0)),
            float(frontier_stagnation_ratio),
            budget_used_ratio,
        ], dtype=np.float32)

    # ---- Elite tracking and shared updates ----

    def _update_elite_frontier(self, population: List[Solution], best: Optional[Solution]) -> int:
        """Update the top-10% elite frontier.

        Strategy:
        - Periodically rebuild the frontier in O(N log K) using a bounded heap (no full sort).
        - Between rebuilds, incrementally attempt to insert only the current best in O(log K).

        Returns:
            1 if the frontier (cutoff) significantly improved under the adaptive gate; else 0.
        """
        if (not self.elites_initialized) or (self.step_index % self.elite_refresh_period == 0):
            improved = self._rebuild_elite_frontier(population)
            self.elites_initialized = True
            return improved
        # Cheap incremental path
        return self._incremental_elite_update_from_best(best)

    def _rebuild_elite_frontier(self, population: List[Solution]) -> int:
        """Rebuild the top-K elite frontier using a bounded heap in O(N log K)."""
        self.elite_heap.clear()
        self.elite_id_set.clear()
        population_size = max(0, len(population))
        elite_size = max(1, int(math.ceil(self.elite_top_fraction * population_size))) if population_size > 0 else 1
        self.elite_size = elite_size
        for sol in population:
            if sol is None or sol.fitness is None:
                continue
            fitness_value = float(sol.fitness)
            sol_id = self._stable_solution_id(sol)
            # Deduplicate IDs within the heap build to keep elites unique when possible
            if sol_id in self.elite_id_set:
                continue
            if len(self.elite_heap) < elite_size:
                heapq.heappush(self.elite_heap, (-fitness_value, sol_id))
                self.elite_id_set.add(sol_id)
            else:
                worst_neg, _ = self.elite_heap[0]
                current_worst = -worst_neg
                if fitness_value < current_worst and sol_id not in self.elite_id_set:
                    popped = heapq.heapreplace(self.elite_heap, (-fitness_value, sol_id))
                    self.elite_id_set.discard(popped[1])
                    self.elite_id_set.add(sol_id)

        self.elite_worst_fitness = -self.elite_heap[0][0] if self.elite_heap else math.inf
        improved = self._update_cutoff_improvement_state()

        # Update entropy snapshot and cache (once per refresh)
        snapshot_ids = list(self.elite_id_set)
        self._push_elite_snapshot(snapshot_ids)
        self.cached_entropy_value = self._compute_elite_entropy()
        return improved

    def _incremental_elite_update_from_best(self, best: Optional[Solution]) -> int:
        """Attempt to update the frontier using only the current best (O(log K))."""
        if best is None or best.fitness is None or self.elite_size <= 0:
            return self._update_cutoff_improvement_state(no_change=True)
        fitness_value = float(best.fitness)
        sol_id = self._stable_solution_id(best)
        # Already an elite → no change
        if sol_id in self.elite_id_set:
            return self._update_cutoff_improvement_state(no_change=True)
        # Insert if it improves the cutoff
        current_worst = -self.elite_heap[0][0] if self.elite_heap else math.inf
        if fitness_value < current_worst:
            popped = heapq.heapreplace(self.elite_heap, (-fitness_value, sol_id)) if self.elite_heap else None
            if popped is not None:
                self.elite_id_set.discard(popped[1])
            self.elite_id_set.add(sol_id)
            self.elite_worst_fitness = -self.elite_heap[0][0]
            return self._update_cutoff_improvement_state()
        return self._update_cutoff_improvement_state(no_change=True)

    def _update_cutoff_improvement_state(self, *, no_change: bool = False) -> int:
        """Update adaptive improvement state based on the frontier cutoff change."""
        prev = self.prev_elite_worst_fitness
        cur = self.elite_worst_fitness
        if no_change or not math.isfinite(prev) or not math.isfinite(cur):
            improved_flag = 0
            abs_delta = 0.0
        else:
            delta = prev - cur  # >0 → improvement (lower worst fitness)
            abs_delta = abs(delta)
            improved_flag = 1 if (delta > 0.0 and abs_delta > (self.improvement_gate_coeff * self.ewma_cutoff_change)) else 0

        # Always update EWMA magnitude
        self.ewma_cutoff_change = (
            self.improvement_ewma_alpha * abs_delta + (1.0 - self.improvement_ewma_alpha) * self.ewma_cutoff_change
        )

        # Update age and windowed success rate
        if improved_flag:
            self.steps_since_elite_improvement = 0
        else:
            self.steps_since_elite_improvement += 1
        if len(self.improvement_flag_window) == self.improvement_flag_window.maxlen:
            self.improvement_flag_sum -= int(self.improvement_flag_window[0])
        self.improvement_flag_window.append(improved_flag)
        self.improvement_flag_sum += int(improved_flag)

        self.prev_elite_worst_fitness = cur
        return improved_flag

    # ---- Entropy helpers ----

    def _push_elite_snapshot(self, ids: List[str]) -> None:
        """Insert a de-duplicated elite ID snapshot into the entropy window."""
        # Evict oldest snapshot
        if len(self.elite_snapshot_window) == self.elite_snapshot_window.maxlen:
            old = self.elite_snapshot_window.popleft()
            for _id in old:
                self.elite_id_counts[_id] -= 1
                if self.elite_id_counts[_id] <= 0:
                    del self.elite_id_counts[_id]
        # Append new snapshot (deduplicated)
        unique_ids = list(set(ids))
        self.elite_snapshot_window.append(unique_ids)
        for _id in unique_ids:
            self.elite_id_counts[_id] += 1

    def _compute_elite_entropy(self) -> float:
        """Compute normalized Shannon entropy over elite ID frequencies in the window."""
        total = sum(self.elite_id_counts.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in self.elite_id_counts.values():
            p = count / total
            if p > 0.0:
                entropy -= p * math.log(p + 1e-18)
        norm = math.log(max(1, self.elite_size * len(self.elite_snapshot_window)))
        if norm <= 0.0:
            return 0.0
        return float(entropy / norm)

    # ---- Utility ----

    def _stable_solution_id(self, sol: Solution) -> str:
        """Produce a stable, jitter-resistant ID for a solution's representation."""
        sol_id_attr = getattr(sol, "id", None)
        if sol_id_attr is not None:
            return str(sol_id_attr)
        rep = sol.representation
        try:
            if hasattr(rep, "tobytes"):
                arr = rep
                # Quantize floats to reduce jitter
                if hasattr(arr, "dtype") and np.issubdtype(arr.dtype, np.floating):
                    arr = np.round(arr.astype(np.float64), 8)
                data = arr.tobytes()
            elif isinstance(rep, (list, tuple)):
                # Quantize floats inside nested lists/tuples
                def _quantize(x):
                    if isinstance(x, float):
                        return round(x, 8)
                    return x
                data = repr(tuple(_quantize(x) for x in rep)).encode("utf-8")
            else:
                data = repr(rep).encode("utf-8")
        except Exception:
            data = str(id(rep)).encode("utf-8")
        return hashlib.blake2b(data, digest_size=12).hexdigest()

    # ---- Bounds extraction ----
    def _extract_bounds(self, meta: dict) -> tuple[float, float]:
        """Derive fitness bounds from a metadata dict.

        Supports keys (first valid pair wins):
        - lower_bound / upper_bound
        - fitness_lower_bound / fitness_upper_bound
        - fitness_min / fitness_max
        Falls back to [0,1] if unavailable. Variable-domain bounds (e.g., per-dimension) are
        intentionally ignored here because this normalization is for scalar fitness values.
        """
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
