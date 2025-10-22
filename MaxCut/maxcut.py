from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from Core.problem import ProblemInterface, Solution


class MaxCutProblem(ProblemInterface):
    """Unweighted (0/1) Maximum Cut posed as a binary minimization problem."""

    def __init__(
        self,
        weight_matrix: Sequence[Sequence[float]],
        *,
        seed: Optional[int] = None,
        ensure_connected: bool = False,
    ) -> None:
        """
        Args:
            weight_matrix: Symmetric 0/1 matrix indicating the presence of edges.
            seed: Optional RNG seed for consistent random initializations.
            ensure_connected: If True, repairs random partitions to keep both sides non-empty.
        """
        weights = np.asarray(weight_matrix, dtype=float)
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError("weight_matrix must be a square matrix")
        if weights.size == 0:
            raise ValueError("weight_matrix must be non-empty")
        if not np.allclose(weights, weights.T, atol=1e-8):
            raise ValueError("weight_matrix must be symmetric for an undirected graph")
        if not np.all(
            np.logical_or(np.isclose(weights, 0.0, atol=1e-8), np.isclose(weights, 1.0, atol=1e-8))
        ):
            raise ValueError("weight_matrix must contain only 0/1 entries for unweighted Max-Cut")

        # Zero the diagonal to ignore self-loops.
        weights = weights.copy()
        np.fill_diagonal(weights, 0.0)
        weights = np.where(weights > 0.5, 1.0, 0.0)

        self.weights = weights
        self.n_nodes = weights.shape[0]
        self.ensure_connected = ensure_connected
        self._rng = np.random.default_rng(seed)

        self._total_weight = float(np.sum(np.triu(self.weights, k=1)))
        self._lower_bounds = np.zeros(self.n_nodes, dtype=float)
        self._upper_bounds = np.ones(self.n_nodes, dtype=float)

    # ---- ProblemInterface API -------------------------------------------------
    def evaluate(self, solution: Solution) -> float:
        mask = self._to_vector(solution.representation)
        cut_val = self.cut_value(mask)
        fitness = -cut_val  # Maximizing cut => minimizing negative cut.
        solution.fitness = fitness
        return fitness

    def get_initial_solution(self) -> Solution:
        mask = self._random_partition()
        sol = Solution(mask.astype(int).tolist(), self)
        sol.evaluate()
        return sol

    def get_initial_population(self, population_size: int) -> list[Solution]:
        pop = []
        for _ in range(max(1, int(population_size))):
            mask = self._random_partition()
            sol = Solution(mask.astype(int).tolist(), self)
            sol.evaluate()
            pop.append(sol)
        return pop

    def get_problem_info(self) -> Dict[str, Any]:
        return {
            "dimension": int(self.n_nodes),
            "problem_type": "binary",
            "lower_bounds": self._lower_bounds.copy(),
            "upper_bounds": self._upper_bounds.copy(),
            "fitness_bounds": (-self._total_weight, 0.0),
            "total_edge_weight": float(self._total_weight),
        }

    # ---- Environment helpers --------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """Re-seeds the RNG and optionally sets a deterministic initial cut."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "initial_partition" in options:
            part = self._to_vector(options["initial_partition"])
            if self.ensure_connected:
                part = self._repair_partition(part)
            options["initial_partition"] = part.tolist()

    # Optional hooks for local search algorithms -------------------------------
    def neighbor(self, solution: Solution) -> Solution:
        mask = self._to_vector(solution.representation)
        idx = int(self._rng.integers(0, self.n_nodes))
        mask[idx] = 1.0 - mask[idx]
        mask = self._repair_partition(mask) if self.ensure_connected else mask
        return Solution(mask.astype(int).tolist(), self)

    def repair(self, representation: Iterable[int]) -> np.ndarray:
        mask = self._to_vector(representation)
        return self._repair_partition(mask)

    # ---- Domain-specific utilities -------------------------------------------
    def cut_value(self, mask: Iterable[int]) -> float:
        vec = self._to_vector(mask)
        mask_bool = vec.astype(bool)
        if mask_bool.size == 0:
            return 0.0
        cut = float(np.sum(self.weights[np.outer(mask_bool, ~mask_bool)]))
        return cut

    # ---- Internal helpers -----------------------------------------------------
    def _to_vector(self, rep: Iterable[int]) -> np.ndarray:
        arr = np.asarray(list(rep), dtype=int)
        if arr.size != self.n_nodes:
            raise ValueError("representation length mismatch with number of nodes")
        arr = np.clip(arr, 0, 1)
        return arr.astype(float)

    def _random_partition(self) -> np.ndarray:
        mask = self._rng.integers(0, 2, size=self.n_nodes).astype(float)
        if self.ensure_connected:
            mask = self._repair_partition(mask)
        return mask

    def _repair_partition(self, mask: np.ndarray) -> np.ndarray:
        mask = np.clip(mask.astype(float, copy=True), 0.0, 1.0)
        if mask.size == 0:
            return mask
        if np.allclose(mask, 0.0) or np.allclose(mask, 1.0):
            idx = int(self._rng.integers(0, self.n_nodes))
            mask[idx] = 1.0 - mask[idx]
        return mask


@dataclass
class MaxCutSpec:
    """Specification for generating random unweighted graphs for Max-Cut."""

    n_nodes: int
    edge_probability: float
    seed: Optional[int] = None


def generate_random_maxcut(spec: MaxCutSpec) -> Tuple[MaxCutProblem, Dict[str, Any]]:
    """
    Generates a random unweighted undirected graph and the corresponding MaxCutProblem.

    Returns:
        (problem, metadata) tuple where metadata includes the generated weights.
    """
    if spec.n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if not (0.0 <= spec.edge_probability <= 1.0):
        raise ValueError("edge_probability must be within [0, 1]")

    rng = np.random.default_rng(spec.seed)
    weights = np.zeros((spec.n_nodes, spec.n_nodes), dtype=float)

    edges = rng.uniform(0.0, 1.0, size=(spec.n_nodes, spec.n_nodes)) < spec.edge_probability
    upper = np.triu(edges, k=1)
    weights[upper] = 1.0
    weights = weights + weights.T  # Symmetrize and keep binary
    weights = np.where(weights > 0.5, 1.0, 0.0)

    problem = MaxCutProblem(weights, seed=spec.seed)

    metadata = {
        "weight_matrix": weights,
        "spec": spec,
        "edge_probability": float(spec.edge_probability),
    }
    return problem, metadata
