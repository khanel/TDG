import numpy as np
from typing import Dict, Iterable, Optional, Tuple
from Core.search_algorithm import SearchAlgorithm
from Core.problem import Solution
from Problem import Problem, Indexer, Elite

class MapElites(SearchAlgorithm):
    def __init__(self, problem, population_size, **kwargs):
        # Initialize with framework compatibility
        super().__init__(problem, population_size, **kwargs)
        self.indexer = kwargs.get('indexer')
        if self.indexer is None:
            raise ValueError("MapElites requires an 'indexer' parameter")
        self.minimize = kwargs.get('minimize', False)
        self.rng = kwargs.get('rng', np.random.default_rng())
        self.archive: Dict[Tuple, Elite] = {}

    def initialize(self):
        """Initialize population and archive."""
        super().initialize()  # This will populate self.population
        # Initialize archive from initial population
        for sol in self.population:
            if hasattr(sol, 'representation'):
                x = sol.representation
            else:
                x = sol  # Assume sol is already the representation
            f = sol.fitness if sol.fitness is not None else self.problem.evaluate(sol)
            bd = self.problem.behavior_descriptor(x)
            key = self.indexer.key(bd)
            if key is not None:
                self._add_to_archive(key, x, f, bd)
        self._update_best_solution()

    def step(self):
        """Perform one MAP-Elites iteration."""
        n_offspring = getattr(self, 'batch_size', 64)  # Default batch size
        parent_selector = getattr(self, 'parent_selector', "uniform")
        crossover_rate = getattr(self, 'crossover_rate', 0.0)

        if not self.archive:
            # Bootstrap if empty
            self.initialize()
            return

        parents = self._select_parents(n_offspring, parent_selector)

        new_population = []
        for parent in parents:
            # Apply crossover if enabled
            if self.rng.random() < crossover_rate and len(self.archive) >= 2:
                parent2 = self._select_parents(1, parent_selector)[0]
                child_x = self.problem.crossover(parent.x, parent2.x, self.rng)
            else:
                child_x = parent.x

            # Apply mutation
            child_x = self.problem.mutate(child_x, self.rng)

            # Evaluate
            f = self.problem.fitness(child_x)
            bd = self.problem.behavior_descriptor(child_x)
            key = self.indexer.key(bd)
            if key is not None:
                self._add_to_archive(key, child_x, f, bd)
                # Create Solution for population
                child_sol = Solution(child_x, self.problem)
                child_sol.fitness = f
                new_population.append(child_sol)

        # Update population for framework compatibility
        if new_population:
            self.population = new_population[:self.population_size]
        self._update_best_solution()

    def _select_parents(self, n: int, selector: str) -> list[Elite]:
        """Select n parents from the archive."""
        if selector == "uniform":
            keys = list(self.archive.keys())
            selected_keys = self.rng.choice(keys, size=n, replace=True)
            return [self.archive[key] for key in selected_keys]
        else:
            # Default to uniform
            return self._select_parents(n, "uniform")

    def _add_to_archive(self, key: Tuple, x: np.ndarray, fitness: float, bd: np.ndarray) -> None:
        """Add solution to archive if it improves the cell."""
        better = (lambda a, b: a < b) if self.minimize else (lambda a, b: a > b)
        current = self.archive.get(key)
        if current is None or better(fitness, current.fitness):
            self.archive[key] = Elite(x, fitness, bd)

    def archive_items(self) -> Iterable[Tuple[Tuple, Elite]]:
        """Return iterator over (key, elite) pairs."""
        return self.archive.items()

    def stats(self) -> dict:
        """Return {coverage, qd_score, best_fitness, filled, total_cells}."""
        if not self.archive:
            return {
                "coverage": 0.0,
                "qd_score": 0.0,
                "best_fitness": None,
                "filled": 0,
                "total_cells": self.indexer.size()
            }

        filled = len(self.archive)
        total_cells = self.indexer.size()
        coverage = filled / total_cells

        # QD score is sum of fitnesses
        qd_score = sum(elite.fitness for elite in self.archive.values())

        # Best fitness
        best_fitness = max(elite.fitness for elite in self.archive.values()) if not self.minimize else min(elite.fitness for elite in self.archive.values())

        return {
            "coverage": coverage,
            "qd_score": qd_score,
            "best_fitness": best_fitness,
            "filled": filled,
            "total_cells": total_cells
        }

    def save(self, path: str) -> None:
        """Save archive to numpy .npz file."""
        keys = list(self.archive.keys())
        fitnesses = [self.archive[k].fitness for k in keys]
        bds = np.array([self.archive[k].bd for k in keys])
        xs = np.array([self.archive[k].x for k in keys])

        np.savez(path, keys=keys, fitnesses=fitnesses, bds=bds, xs=xs)

    def load(self, path: str) -> None:
        """Load archive from numpy .npz file."""
        data = np.load(path, allow_pickle=True)
        keys = data['keys']
        fitnesses = data['fitnesses']
        bds = data['bds']
        xs = data['xs']

        self.archive.clear()
        for key, fitness, bd, x in zip(keys, fitnesses, bds, xs):
            self.archive[tuple(key)] = Elite(x, fitness, bd)

    def run(self, max_iterations: int = 1000) -> Optional[Solution]:
        """Run the algorithm for max_iterations steps (framework compatibility)."""
        for _ in range(max_iterations):
            self.step()
        return self.get_best_solution()