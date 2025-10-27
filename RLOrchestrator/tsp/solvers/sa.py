"""
TSP Simulated Annealing solver for exploitation phase.
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm
from SA.SA import SimulatedAnnealing


class TSPSimulatedAnnealing(SearchAlgorithm):
    """Simulated Annealing tailored for TSP exploitation."""

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 1,
        *,
        initial_temperature: float = 10.0,
        final_temperature: float = 1e-3,
        cooling_rate: float = 0.95,
        moves_per_temp: int = 10,
        max_iterations: int = 1000,
        neighbor_fn: Optional[Callable[[Solution], Solution]] = None,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "tsp_problem"):
            raise ValueError("TSPSimulatedAnnealing expects a TSPAdapter exposing `tsp_problem`.")
        super().__init__(problem, population_size)
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.moves_per_temp = moves_per_temp
        self.max_iterations = max_iterations
        self.neighbor_fn = neighbor_fn
        self.rng = np.random.default_rng(seed)
        self.num_cities = len(problem.tsp_problem.city_coords)
        if self.num_cities < 3:
            raise ValueError("TSPSimulatedAnnealing expects at least 3 cities.")

        # Create the underlying SA instance
        self.sa = SimulatedAnnealing(
            problem=problem,
            initial_temperature=initial_temperature,
            final_temperature=final_temperature,
            cooling_rate=cooling_rate,
            moves_per_temp=moves_per_temp,
            max_iterations=max_iterations,
            neighbor_fn=neighbor_fn or self._tsp_neighbor,
            population_size=population_size,
        )

    def initialize(self):
        """
        Initializes the underlying SA solver and syncs the state.
        """ 
        self.sa.initialize()
        self.population = self.sa.population.copy()
        self.best_solution = self.sa.best_solution.copy() if self.sa.best_solution else None
        self.iteration = 0
        # Prepare distance matrix and embedding support
        self._refresh_distance_cache()
        self._update_population_matrix()

    def reset(self):
        """Reset the state of the SA solver for a new run."""
        self.sa.temperature = self.initial_temperature
        self.sa.iteration = 0
        self.sa.best_solution = None
        self.sa.current_solution = None

    def step(self):
        """
        Performs one step of the SA algorithm, ensuring it starts from the best
        solution provided by the orchestrator.
        """
        if not self.population:
            return

        best_seed = min(self.population, key=lambda s: s.fitness if s and s.fitness is not None else float('inf'))

        if self.sa.current_solution is None:
            self.sa.current_solution = best_seed.copy()
            self.sa.best_solution = best_seed.copy()

        self.sa.step()

        self.population = self.sa.get_population()
        self.best_solution = self.sa.get_best_solution()
        self.iteration += 1
        # Maintain permutation-aware embedding
        self._update_population_matrix()

    def ingest_seeds(self, seeds: list[Solution]) -> None:
        """Synchronize the internal SA state with externally provided seeds."""
        self.sa.ingest_seeds(seeds)
        if not self.sa.population:
            return

        self.population = [sol.copy(preserve_id=False) for sol in self.sa.population]
        current_best = self.sa.get_best_solution()
        self.best_solution = current_best.copy() if current_best else None
        self.iteration = self.sa.iteration
        self._refresh_distance_cache()
        self._update_population_matrix()

    def _tsp_neighbor(self, sol: Solution) -> Solution:
        """TSP-specific neighbor function: 2-opt or swap moves."""
        tour = list(sol.representation)
        if len(tour) < 4:
            return sol.copy()

        # Choose between 2-opt and swap
        if self.rng.random() < 0.5:
            # 2-opt: reverse a segment
            i, j = sorted(self.rng.choice(range(1, len(tour)), size=2, replace=False))
            new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        else:
            # Swap: exchange two cities
            i, j = self.rng.choice(range(1, len(tour)), size=2, replace=False)
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        # Ensure city 1 stays at start
        if new_tour[0] != 1:
            idx = new_tour.index(1)
            new_tour[0], new_tour[idx] = new_tour[idx], new_tour[0]

        return Solution(new_tour, self.problem)

    # --- Observation support: permutation-aware embedding ----------------
    def _refresh_distance_cache(self):
        tsp = self.problem.tsp_problem
        dm = np.asarray(tsp.cities_graph.get_weights(), dtype=float)
        self._distance_matrix = dm
        self._num_cities = dm.shape[0]
        max_edge = float(np.max(dm)) if dm.size else 1.0
        self._hist_bins = np.linspace(0.0, max(1e-9, max_edge), num=17)

    def _tour_edge_lengths(self, rep: list[int]) -> np.ndarray:
        idx = [i - 1 for i in rep]
        idx.append(idx[0])
        dm = self._distance_matrix
        edges = [dm[idx[i], idx[i+1]] for i in range(self._num_cities)]
        return np.asarray(edges, dtype=float)

    def _tour_hist_vector(self, rep: list[int]) -> np.ndarray:
        edges = self._tour_edge_lengths(rep)
        hist, _ = np.histogram(edges, bins=self._hist_bins, density=False)
        hist = hist.astype(float)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist.astype(np.float32)

    def _update_population_matrix(self) -> None:
        try:
            if not self.population:
                self._population_matrix = None
                return
            feats = [self._tour_hist_vector(list(sol.representation)) for sol in self.population]
            self._population_matrix = np.stack(feats, axis=0)
        except Exception:
            self._population_matrix = None
