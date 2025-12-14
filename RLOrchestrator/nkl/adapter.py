"""NK-Landscape problem adapter with optional per-episode randomization."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Core.problem import ProblemInterface, Solution
from problems.NKL.nkl import NKLProblem


class NKLAdapter(ProblemInterface):
    """Adapter around :class:`NKLProblem` with regeneration hooks."""

    def __init__(
        self,
        n_items: int = 100,
        k_interactions: int = 5,
        seed: Optional[int] = 42,
    ):
        self._rng = np.random.default_rng(seed)
        self._n_items_range: Optional[Tuple[int, int]] = None
        self._k_interactions_range: Optional[Tuple[int, int]] = None

        if isinstance(n_items, (tuple, list)) and len(n_items) == 2:
            self._n_items_range = (max(2, n_items[0]), max(2, n_items[1]))
        else:
            self._n_items_range = (max(2, n_items), max(2, n_items))

        if isinstance(k_interactions, (tuple, list)) and len(k_interactions) == 2:
            self._k_interactions_range = (max(0, k_interactions[0]), max(0, k_interactions[1]))
        else:
            self._k_interactions_range = (max(0, k_interactions), max(0, k_interactions))

        self.nkl_problem = self._create_problem_instance()
        self._update_bounds()

    def _create_problem_instance(self) -> NKLProblem:
        n = self._rng.integers(self._n_items_range[0], self._n_items_range[1] + 1)
        max_k = n - 1
        k_min = min(max_k, self._k_interactions_range[0])
        k_max = min(max_k, self._k_interactions_range[1])
        k = self._rng.integers(k_min, k_max + 1) if k_min < k_max else k_min
        
        seed = int(self._rng.integers(0, 2**31))
        return NKLProblem(n=n, k=k, seed=seed)

    def evaluate(self, solution: Solution) -> float:
        # Coerce representation to a 1D binary vector with current dimension
        rep = np.asarray(solution.representation)
        if rep.ndim != 1:
            rep = rep.reshape(-1)
        # Ensure binary values {0,1}
        rep = np.where(rep > 0, 1, 0).astype(int)

        n = int(self.nkl_problem.n)
        m = int(rep.shape[0])
        if m != n:
            # Align length with current problem dimension
            if m > n:
                rep = rep[:n]
            else:
                # Pad with random bits for robustness when dimension increased
                pad = self._rng.integers(0, 2, size=(n - m,), dtype=int)
                rep = np.concatenate([rep, pad])

        fitness = self.nkl_problem.evaluate(rep)
        solution.fitness = fitness
        return fitness

    def batch_evaluate_solutions(self, solutions: List[Solution]) -> np.ndarray:
        """
        Highly optimized vectorized batch evaluation for multiple Solution objects.
        
        Args:
            solutions: List of Solution objects
            
        Returns:
            Array of fitness values
        """
        if not solutions:
            return np.array([])
        
        n = int(self.nkl_problem.n)
        batch_size = len(solutions)
        
        # Pre-allocate array for better performance
        representations = np.zeros((batch_size, n), dtype=int)
        
        for i, sol in enumerate(solutions):
            rep = np.asarray(sol.representation)
            if rep.ndim != 1:
                rep = rep.reshape(-1)
            # Ensure binary values {0,1}
            rep = np.where(rep > 0, 1, 0).astype(int)
            
            m = int(rep.shape[0])
            if m != n:
                # Handle size mismatch
                if m > n:
                    representations[i] = rep[:n]
                else:
                    representations[i, :m] = rep
                    # Pad with zeros (deterministic for batch consistency)
            else:
                representations[i] = rep
        
        # Vectorized batch evaluation using core's evaluate_batch
        fitnesses = self.nkl_problem.evaluate_batch(representations)
        
        # Update solution fitnesses in-place
        for sol, fitness in zip(solutions, fitnesses):
            sol.fitness = float(fitness)
        
        return fitnesses

    def get_initial_solution(self) -> Solution:
        n = self.nkl_problem.n
        mask = self._rng.integers(0, 2, size=n, dtype=int)
        sol = Solution(mask.tolist(), self)
        sol.evaluate()
        return sol

    def get_initial_population(self, size: int) -> List[Solution]:
        """
        Generates an initial population of solutions using vectorized evaluation.
        """
        n = self.nkl_problem.n
        population_size = max(1, int(size))
        
        # Vectorized population generation
        population_representations = self._rng.integers(0, 2, size=(population_size, n), dtype=int)
        
        # Create solution objects
        population = [Solution(rep.tolist(), self) for rep in population_representations]
        
        # Use vectorized batch evaluation for efficiency
        self.batch_evaluate_solutions(population)
        
        return population

    def get_problem_info(self) -> Dict[str, Any]:
        return {
            "dimension": self.nkl_problem.n,
            "problem_type": "binary",
            "lower_bounds": np.zeros(self.nkl_problem.n, dtype=float),
            "upper_bounds": np.ones(self.nkl_problem.n, dtype=float),
        }

    def get_bounds(self) -> Dict[str, float]:
        return dict(self._bounds)

    def regenerate_instance(self) -> bool:
        self.nkl_problem = self._create_problem_instance()
        self._update_bounds()
        return True

    def _update_bounds(self) -> None:
        # For NKL, fitness is the negative average of values in [0, 1].
        # So, the fitness range is [-1, 0].
        self._bounds = {"lower_bound": -1.0, "upper_bound": 0.0}

    def repair_mask(self, mask: list[int]) -> np.ndarray:
        # NKL is an unconstrained binary problem, so no repair is needed.
        return np.asarray(mask, dtype=int)
