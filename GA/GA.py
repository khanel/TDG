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

            # Crossover and flatten the offspring list
            offspring = []
            for _ in range(self.population_size // 2):
                children = self.genetic_operator.crossover(parents[0], parents[1])
                offspring.extend(children)

            # Mutation
            offspring = [self.genetic_operator.mutate(individual) for individual in offspring]

            # Replacement
            population = self._replace_least_fit(population, offspring)

            # Update best solution
            current_best = min(population, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
            if current_best.fitness is None:
                current_best.evaluate()
            current_best_fitness = current_best.fitness

            # Check if current best is better than overall best
            if current_best_fitness < best_fitness:
                best_solution = current_best
                best_fitness = current_best_fitness

            # Log statistics if available
            if hasattr(self.problem, 'log_statistics'):
                self.problem.log_statistics(population, iteration)

            # Verbosity handling
            if self.verbosity >= 1:
                print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")

        return best_solution, best_fitness

    def _replace_least_fit(self, population, offspring):
        # Ensure all individuals have fitness calculated
        for ind in population + offspring:
            if ind.fitness is None:
                ind.evaluate()
        
        # Sort population by fitness
        population = sorted(population, key=lambda x: x.fitness)
        offspring = sorted(offspring, key=lambda x: x.fitness)
        
        # Keep the best individuals from both population and offspring
        new_population = population[:self.population_size - len(offspring)] + offspring
        return new_population
