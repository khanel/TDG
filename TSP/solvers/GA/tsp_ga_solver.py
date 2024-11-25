# Built-in Python packages
import sys
from pathlib import Path
import time

# Third-party packages
import numpy as np
import matplotlib.pyplot as plt

# Local project modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from GA.GA import GeneticAlgorithm
from GA.Problem import ProblemInterface, Solution, GeneticOperator
from TSP.TSP import TSPProblem, Graph

class TSPGeneticOperator(GeneticOperator):
    def __init__(self, selection_prob=0.8, mutation_prob=0.1):
        self.selection_prob = selection_prob
        self.mutation_prob = mutation_prob
    
    def select(self, population):
        # Tournament selection
        tournament_size = 3
        selected = []
        for _ in range(2):
            tournament = np.random.choice(population, size=tournament_size, replace=False)
            winner = min(tournament, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
            if winner.fitness is None:
                winner.evaluate()
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        if np.random.random() > self.selection_prob:
            return [parent1, parent2]

        # Order Crossover (OX) maintaining city 1 as start
        n = len(parent1.representation)
        # Choose section after city 1
        start = 1 + np.random.randint(0, n-2)
        end = 1 + np.random.randint(start, n-1)
        
        def create_child(p1, p2):
            # Keep city 1 as start, apply OX to rest of route
            child = [1] + [-1] * (n-1)  # Initialize with -1s after city 1
            # Copy section from parent1
            child[start:end+1] = p1.representation[start:end+1]
            # Fill remaining positions with cities from parent2 in order
            remaining = [x for x in p2.representation if x not in child[start:end+1] and x != 1]
            # Fill positions after end
            for i in range(end+1, n):
                child[i] = remaining.pop(0)
            # Fill positions between city 1 and start
            for i in range(1, start):
                child[i] = remaining.pop(0)
            return Solution(child, parent1.problem)

        child1 = create_child(parent1, parent2)
        child2 = create_child(parent2, parent1)
        child1.evaluate()
        child2.evaluate()
        return [child1, child2]

    def mutate(self, individual):
        if np.random.random() > self.mutation_prob:
            return individual
        
        # Swap Mutation (avoiding city 1)
        n = len(individual.representation)
        # Choose two random positions after city 1
        pos1, pos2 = np.random.choice(range(1, n), size=2, replace=False)
        
        # Create new representation with the swap
        new_repr = individual.representation.copy()
        new_repr[pos1], new_repr[pos2] = new_repr[pos2], new_repr[pos1]
        
        mutated = Solution(new_repr, individual.problem)
        mutated.evaluate()
        return mutated


class TSPGASolver:
    def __init__(self, tsp_problem, population_size=100, max_iterations=800,
                 selection_prob=0.8, mutation_prob=0.1):
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.genetic_operator = TSPGeneticOperator(selection_prob, mutation_prob)
        
    def solve(self):
        """Run the genetic algorithm to solve the TSP problem."""
        ga = GeneticAlgorithm(
            problem_interface=self.tsp_problem,
            genetic_operator=self.genetic_operator,
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            desired_fitness=0,  # Run for all iterations
            verbosity=0  # Use custom logging
        )
        
        best_solution, best_fitness = ga.run()
        return best_solution, best_fitness


if __name__ == "__main__":
    # Create a sample TSP problem with 20 cities randomly placed in a 100x100 grid
    num_cities = 20
    np.random.seed(42)  # For reproducibility
    city_coords = np.random.rand(num_cities, 2) * 100
    
    # Create distance matrix
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                # Calculate Euclidean distance between cities
                distances[i,j] = np.sqrt(np.sum((city_coords[i] - city_coords[j])**2))
    
    # Create graph and TSP problem instance
    graph = Graph(distances)
    tsp_problem = TSPProblem(graph, city_coords)
    
    # Create and configure the GA solver
    solver = TSPGASolver(
        tsp_problem=tsp_problem,
        population_size=100,
        max_iterations=500,
        selection_prob=0.8,
        mutation_prob=0.1
    )
    
    # Solve the problem
    print("Starting TSP solution with Genetic Algorithm...")
    print(f"Number of cities: {num_cities}")
    print("Initial setup complete. Beginning evolution...")
    
    start_time = time.time()
    best_solution, best_fitness = solver.solve()  # Unpack both return values
    end_time = time.time()
    
    # Print results
    print("\nOptimization complete!")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Best route found has length: {best_fitness:.2f}")
    print("Route:", " -> ".join(map(str, best_solution.representation + [best_solution.representation[0]])))
    
    # Plot the final route and optimization progress
    tsp_problem.plot_route(best_solution)
    tsp_problem.plot_progress()
    
    # Keep the plot window open
    plt.ioff()
    plt.show()