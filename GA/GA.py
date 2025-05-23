import numpy as np
from Core.search_algorithm import SearchAlgorithm
from Core.problem import Solution

class GeneticAlgorithm(SearchAlgorithm):
    def __init__(self, problem, population_size, genetic_operator, max_iterations=100, desired_fitness=0, verbosity=0, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.genetic_operator = genetic_operator
        self.max_iterations = max_iterations
        self.desired_fitness = desired_fitness
        self.verbosity = verbosity
        self.iteration = 0

    def step(self):
        # Selection
        parents = self.genetic_operator.select(self.population)

        # Crossover and flatten the offspring list
        offspring = []
        for _ in range(self.population_size // 2):
            children = self.genetic_operator.crossover(parents[0], parents[1])
            offspring.extend(children)

        # Mutation
        offspring = [self.genetic_operator.mutate(individual) for individual in offspring]

        # Replacement
        self.population = self._replace_least_fit(self.population, offspring)

        # Update best solution
        self._update_best_solution()
        self.iteration += 1

        # Log statistics if available
        if hasattr(self.problem, 'log_statistics'):
            self.problem.log_statistics(self.population, self.iteration)

        # Verbosity handling
        if self.verbosity >= 1:
            print(f"Iteration {self.iteration}, Best Fitness: {self.best_solution.fitness}")

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
