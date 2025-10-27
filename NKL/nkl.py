"""
Core implementation of the NK-Landscape problem generator.

This version is fully vectorized for high-performance evaluation.
"""

import numpy as np


class NKLProblem:
    """
    Generates and evaluates NK-Landscape problems using a fully vectorized approach.
    """

    def __init__(self, n: int, k: int, seed: int = None):
        """
        Initializes a high-performance NK-Landscape problem instance.

        Args:
            n: The number of decision variables (genes).
            k: The number of epistatic interactions for each gene.
            seed: An optional seed for the random number generator.
        """
        if not (0 <= k < n):
            raise ValueError("K must be in the range [0, N-1]")

        self.n = n
        self.k = k
        self.rng = np.random.default_rng(seed)

        # --- Pre-compute structures for vectorized evaluation ---

        # 1. Epistatic connections for each gene.
        self._connections = self._generate_connections()

        # 2. Full (N, K+1) array of indices needed for evaluation.
        # self._all_indices[i] contains the indices for gene i's fitness calc.
        self._all_indices = np.zeros((n, k + 1), dtype=int)
        self._all_indices[:, 0] = np.arange(n)
        self._all_indices[:, 1:] = self._connections

        # 3. Powers of 2 for fast binary-to-integer conversion.
        self._powers_of_2 = 1 << np.arange(k, -1, -1)

        # 4. Random fitness contribution lookup tables.
        num_entries_per_table = 2 ** (k + 1)
        self.tables = self.rng.random((n, num_entries_per_table))
        
        # 5. An array [0, 1, 2, ..., N-1] for advanced indexing.
        self._row_indices = np.arange(n)

    def _generate_connections(self) -> np.ndarray:
        """Generates random epistatic connections, ensuring no self-connections."""
        connections = np.zeros((self.n, self.k), dtype=int)
        for i in range(self.n):
            possible_neighbors = list(range(self.n))
            possible_neighbors.remove(i)
            connections[i] = self.rng.choice(possible_neighbors, self.k, replace=False)
        return connections

    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluates the fitness of a single solution by wrapping the batch evaluator.
        """
        if solution.ndim != 1:
            raise ValueError("Input to `evaluate` must be a 1D array.")
        
        # Add a batch dimension and call the batch evaluator
        fitness = self.evaluate_batch(solution[np.newaxis, :])
        return fitness[0]

    def evaluate_batch(self, solutions: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitness of a batch of solutions using a fully vectorized method.

        Args:
            solutions: A 2D numpy array of shape (num_solutions, N).

        Returns:
            A 1D numpy array of fitness values for each solution.
        """
        if solutions.ndim != 2 or solutions.shape[1] != self.n:
            raise ValueError(f"Solutions must be a 2D numpy array of shape (num_solutions, {self.n})")

        # 1. Gather all gene patterns for all solutions at once.
        # Shape: (num_solutions, N, K+1)
        patterns = solutions[:, self._all_indices]

        # 2. Convert all patterns to indices in one vectorized operation.
        # Shape: (num_solutions, N)
        indices = patterns.dot(self._powers_of_2)

        # 3. Look up all fitness contributions at once using advanced indexing.
        # This gets tables[0, indices[:, 0]], tables[1, indices[:, 1]], etc.
        # for each solution in the batch.
        contributions = self.tables[self._row_indices, indices]

        # 4. Sum contributions for each solution and normalize.
        total_fitness = np.sum(contributions, axis=1)

        # Return the negative average fitness for minimization for each solution.
        return -(total_fitness / self.n)