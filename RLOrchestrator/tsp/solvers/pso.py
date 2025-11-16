"""
Permutation-aware PSO tailored for TSP exploitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm


@dataclass
class ParticleState:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    solution: Solution


class TSPParticleSwarm(SearchAlgorithm):
    """Random-keys PSO for TSP permutations."""
    phase = 'exploitation'

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 32,
        *,
        omega: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        vmax: float = 0.5,
        seed: Optional[int] = None,
    ):
        if not hasattr(problem, "tsp_problem"):
            raise ValueError("TSPParticleSwarm expects a TSPAdapter exposing `tsp_problem`.")
        super().__init__(problem, population_size)
        self.omega = float(omega)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.vmax = float(max(1e-6, vmax))
        self.rng = np.random.default_rng(seed)
        self.num_cities = len(problem.tsp_problem.city_coords)
        if self.num_cities < 3:
            raise ValueError("TSPParticleSwarm expects at least 3 cities.")
        self.particles: List[ParticleState] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = float("inf")

    def initialize(self):
        self.num_cities = len(self.problem.tsp_problem.city_coords)
        if self.num_cities < 3:
            raise ValueError("TSPParticleSwarm expects at least 3 cities.")
        self.global_best_position = None
        self.global_best_fitness = float("inf")
        self.population = []
        self.particles = []
        self.best_solution = None
        self.iteration = 0
        for _ in range(self.population_size):
            position = self.rng.random(self.num_cities)
            velocity = self.rng.uniform(-self.vmax, self.vmax, size=self.num_cities)
            sol = self._solution_from_position(position)
            fitness = sol.evaluate()
            particle = ParticleState(position=position, velocity=velocity, best_position=position.copy(), best_fitness=fitness, solution=sol)
            self.particles.append(particle)
            self.population.append(sol)
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = position.copy()
        self._update_best_solution()

    def step(self):
        if not self.particles:
            self.initialize()
        assert self.global_best_position is not None
        for particle in self.particles:
            r1 = self.rng.random(self.num_cities)
            r2 = self.rng.random(self.num_cities)
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best_position - particle.position)
            particle.velocity = np.clip(self.omega * particle.velocity + cognitive + social, -self.vmax, self.vmax)
            particle.position = np.clip(particle.position + particle.velocity, 0.0, 1.0)
            particle.solution = self._solution_from_position(particle.position)
            particle.solution.evaluate()
            if particle.solution.fitness is not None and particle.solution.fitness < particle.best_fitness:
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle.solution.fitness
            if particle.solution.fitness is not None and particle.solution.fitness < self.global_best_fitness:
                self.global_best_position = particle.position.copy()
                self.global_best_fitness = particle.solution.fitness
        self.population = [p.solution.copy(preserve_id=False) for p in self.particles]
        self._update_best_solution()
        self.iteration += 1

    def _solution_from_position(self, position: np.ndarray) -> Solution:
        indices = np.argsort(position)
        zero_pos = np.where(indices == 0)[0][0]
        indices = np.roll(indices, -zero_pos)
        tour = (indices + 1).tolist()
        return Solution(tour, self.problem)
