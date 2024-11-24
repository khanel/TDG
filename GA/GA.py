from random import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, problem_interface, genetic_operator, population_size, max_iterations, desired_fitness, verbosity=0):
        self.problem = problem_interface
        self.genetic_operator = genetic_operator
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.desired_fitness = desired_fitness
        self.verbosity = verbosity

    def run(self):
        population = self.problem.generate_initial_population(self.population_size)
        best_solution = None
        best_fitness = float('inf')

        for iteration in range(self.max_iterations):
            # Selection
            parents = self.genetic_operator.select(population)

            # Crossover
            offspring = [self.genetic_operator.crossover(parents[0], parents[1]) for _ in range(self.population_size // 2)]

            # Mutation
            offspring = [self.genetic_operator.mutate(individual) for individual in offspring]

            # Replacement
            population = self._replace_least_fit(population, offspring)

            # Update best solution
            current_best = min(population, key=lambda x: self.problem.calculate_fitness(x))
            current_best_fitness = self.problem.calculate_fitness(current_best)

            # Check if current best is better than overall best
            if current_best_fitness < best_fitness:
                best_solution = current_best
                best_fitness = current_best_fitness

            # Verbosity handling
            if self.verbosity >= 1:
                print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")

        return best_solution, best_fitness

    def _replace_least_fit(self, population, offspring):
        for individual in offspring:
            least_fit_index = min(enumerate(population), key=lambda x: self.problem.calculate_fitness(x[1]))[0]
            if self.problem.calculate_fitness(individual) < self.problem.calculate_fitness(population[least_fit_index]):
                population[least_fit_index] = individual
        return population
