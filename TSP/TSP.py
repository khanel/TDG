import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

# Add parent directory to Python path to allow imports from GA module
sys.path.append(str(Path(__file__).parent.parent))
from GA.GA import GeneticAlgorithm
from GA.Problem import ProblemInterface, Solution, GeneticOperator

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
            winner = min(tournament, key=lambda x: x.problem.calculate_fitness(x.representation))
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        if np.random.random() > self.selection_prob:
            return parent1, parent2

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
            return child

        child1 = create_child(parent1, parent2)
        child2 = create_child(parent2, parent1)
        
        return Solution(child1, parent1.problem), Solution(child2, parent1.problem)

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
        
        return Solution(new_repr, individual.problem)


# TSP Problem Class
class TSPProblem(ProblemInterface):
    def __init__(self, graph, city_coords):
        self.cities_graph = graph
        self.city_coords = city_coords
        self.path = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.improvement_count = 0
        self.last_best_fitness = float('inf')
        self.generations_without_improvement = 0
        self.live_fig = None
        self.live_ax = None
        
    def setup_live_plot(self):
        plt.ion()  # Turn on interactive mode
        self.live_fig, self.live_ax = plt.subplots(figsize=(10, 10))
        self.live_ax.grid(True)
        self.live_ax.set_title('TSP Route (Live)')
        plt.show()
    
    def update_live_plot(self, solution, generation):
        if self.live_ax is None:
            self.setup_live_plot()
            
        self.live_ax.clear()
        
        # Plot cities
        for i, (x, y) in enumerate(self.city_coords, 1):
            self.live_ax.plot(x, y, 'ro', markersize=10)
            self.live_ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot path with arrows
        path = solution.representation
        n = len(path)
        for i in range(len(path)):
            city1 = path[i]
            city2 = path[(i + 1) % n]
            x1, y1 = self.city_coords[city1-1]
            x2, y2 = self.city_coords[city2-1]
            
            # Calculate arrow properties
            dx = x2 - x1
            dy = y2 - y1
            
            # Draw arrow
            self.live_ax.arrow(x1, y1, dx, dy,
                             head_width=0.05,
                             head_length=0.08,
                             fc='b',
                             ec='b',
                             alpha=0.6,
                             length_includes_head=True)
        
        # Plot starting city
        start_x, start_y = self.city_coords[path[0]-1]
        self.live_ax.plot(start_x, start_y, 'go', markersize=15, alpha=0.5, label='Start')
        
        self.live_ax.set_title(f'Generation {generation}\nCurrent Distance: {solution.fitness:.2f}')
        self.live_ax.axis('equal')
        self.live_ax.grid(True)
        self.live_ax.legend()
        
        # Update the plot
        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        
        # Small delay to make the visualization visible
        plt.pause(0.01)
    
    def log_statistics(self, population, generation):
        # Ensure all individuals have fitness calculated
        for ind in population:
            if ind.fitness is None:
                ind.evaluate()
        fitnesses = [ind.fitness for ind in population]
        current_best = min(fitnesses)
        current_avg = np.mean(fitnesses)
        
        self.best_fitness_history.append(current_best)
        self.avg_fitness_history.append(current_avg)
        
        # Get the best solution in current population
        best_solution = min(population, key=lambda x: x.fitness)
        
        # Update live plot
        self.update_live_plot(best_solution, generation)
        
        # Track improvements
        if current_best < self.last_best_fitness:
            improvement = self.last_best_fitness - current_best
            self.improvement_count += 1
            print(f"\nGeneration {generation}: New best distance: {current_best:.2f} "
                  f"(Improved by: {improvement:.2f}, "
                  f"Total improvements: {self.improvement_count})")
            self.last_best_fitness = current_best
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
            if self.generations_without_improvement % 20 == 0:
                print(f"\nNo improvement for {self.generations_without_improvement} generations. "
                      f"Current best: {current_best:.2f}")

    def generate_initial_population(self, size):
        population = []
        vertices = self.cities_graph.get_vertices()
        for _ in range(size):
            # Always start with city 1, shuffle the rest
            remaining_cities = vertices[1:]  # All cities except city 1
            np.random.shuffle(remaining_cities)
            individual = [1] + remaining_cities  # City 1 is always first
            solution = Solution(individual, self)
            solution.evaluate()
            population.append(solution)
        return population

    def calculate_fitness(self, individual_representation):
        return self.calculate_path_distance(individual_representation)

    def validate_individual(self, individual):
        return len(individual.representation) == len(self.cities_graph.get_vertices()) and len(
            set(individual.representation)) == len(individual.representation)

    def get_individual_size(self):
        return len(self.cities_graph.get_vertices())

    def calculate_path_distance(self, path):
        distance = 0
        path_for_distance = path + [path[0]]  # Always close the loop
        for i in range(len(path_for_distance) - 1):
            distance += self.cities_graph.get_weights()[path_for_distance[i] - 1][path_for_distance[i + 1] - 1]
        return distance

    def plot_progress(self):
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Distance History
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, 'b-', label='Best Distance', linewidth=2)
        plt.plot(self.avg_fitness_history, 'r--', label='Average Distance', alpha=0.5)
        plt.xlabel('Generation')
        plt.ylabel('Total Distance')
        plt.title('Distance Optimization Progress')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Improvement Distribution
        plt.subplot(1, 2, 2)
        # Calculate improvements (negative since we're minimizing)
        improvements = np.array(self.best_fitness_history[:-1]) - np.array(self.best_fitness_history[1:])
        if len(improvements) > 0:
            plt.hist(improvements[improvements > 0], bins=min(20, len(improvements)), 
                    color='green', alpha=0.6)
            plt.xlabel('Distance Reduction')
            plt.ylabel('Frequency')
            plt.title('Distribution of Distance Improvements')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_route(self, solution):
        if solution.fitness is None:
            solution.evaluate()
        
        path = solution.representation
        n = len(path)
        
        plt.figure(figsize=(10, 10))
        
        # Plot cities using actual coordinates
        for i, (x, y) in enumerate(self.city_coords, 1):
            plt.plot(x, y, 'ro', markersize=10)
            plt.annotate(f'City {i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot path
        for i in range(len(path)):
            city1 = path[i]
            city2 = path[(i + 1) % n]
            x1, y1 = self.city_coords[city1-1]  # -1 because cities are numbered from 1
            x2, y2 = self.city_coords[city2-1]
            plt.plot([x1, x2], [y1, y2], 'b-', alpha=0.6)
        
        # Plot starting city with different color/style
        start_x, start_y = self.city_coords[path[0]-1]
        plt.plot(start_x, start_y, 'go', markersize=15, alpha=0.5, label='Start')
        
        plt.title(f'TSP Route (Distance: {solution.fitness:.2f})')
        plt.axis('equal')  # Make the plot circular
        plt.grid(True)
        plt.legend()
        plt.show()

    def create_solution(self, representation):
        return Solution(representation, self)

    def update_path(self, new_path):
        self.path = new_path


# Graph Class for TSP cities representation
class Graph:
    def __init__(self, weights):
        self.weights = weights
        self.vertices = [i + 1 for i in range(len(weights))]
        self.edges = []
        self.calculate_edges(weights)

    def calculate_edges(self, weights, first_index=0):
        for i in range(first_index, len(self.vertices)):
            for j in range(len(self.vertices)):
                if i != j and weights[i][j] != 0:
                    self.edges.append((i + 1, j + 1, weights[i][j]))
        return self.edges

    def add_edge(self, weight):
        self.vertices.append(len(self.vertices) + 1)
        for i, j in enumerate(weight):
            if weight[i] != 0:
                self.edges.append((i + 1, len(self.vertices), weight[i]))
                self.edges.append((len(self.vertices), i + 1, weight[i]))
        return self.edges

    def get_weights(self):
        return self.weights

    def get_vertices(self):
        return self.vertices


# Genetic Algorithm Class
class GeneticAlgorithm:
    def __init__(self, problem_interface, genetic_operator,
                 population_size, max_iterations, desired_fitness, verbosity=0):
        self.problem = problem_interface
        self.genetic_operator = genetic_operator
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.desired_fitness = desired_fitness
        self.verbosity = verbosity

    def run(self):
        population = self.problem.generate_initial_population(self.population_size)

        for generation in range(self.max_iterations):
            # Selection
            parents = self.genetic_operator.select(population)

            # Crossover
            offspring = []
            for _ in range(self.population_size // 2):
                child1, child2 = self.genetic_operator.crossover(parents[0], parents[1])
                offspring.append(child1)
                offspring.append(child2)
                parents = self.genetic_operator.select(population)

            # Mutation
            offspring = [self.genetic_operator.mutate(individual) for individual in offspring]

            # Replacement
            population = self._replace_least_fit(population, offspring)

            self.problem.log_statistics(population, generation)

        # Return the best solution (minimum distance for TSP)
        best_solution = min(population, key=lambda x: x.fitness)
        return best_solution, best_solution.fitness

    def _replace_least_fit(self, population, offspring):
        for individual in offspring:
            individual.evaluate()
            # Find the worst individual (highest distance) to replace
            worst_fit_index = max(enumerate(population), key=lambda x: x[1].fitness)[0]
            if individual.fitness < population[worst_fit_index].fitness:
                population[worst_fit_index] = individual
        return population


# Example Usage
if __name__ == "__main__":
    # Create cities in a circle for a known optimal solution
    num_cities = 20
    radius = 1.0
    
    # Generate city coordinates on a circle
    angles = np.linspace(0, 2*np.pi, num_cities, endpoint=False)
    city_coords = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    
    # Calculate distances between cities
    weights = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x1, y1 = city_coords[i]
                x2, y2 = city_coords[j]
                weights[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2) * 100  # Scale distances for better readability
    
    # The optimal solution is to visit cities in order around the circle
    optimal_route = list(range(1, num_cities + 1))  # Cities numbered from 1 to num_cities
    optimal_distance = sum(weights[i-1][i % num_cities] for i in range(num_cities))  # Include return to start
    
    print("City Coordinates:")
    for i, (x, y) in enumerate(city_coords, 1):
        print(f"City {i}: ({x:.2f}, {y:.2f})")
    
    print("\nOptimal Solution:")
    print(f"Route: {' -> '.join(map(str, optimal_route + [optimal_route[0]]))}") # Show return to start
    print(f"Distance: {optimal_distance:.2f}")
    
    print("\nStarting Genetic Algorithm optimization...")
    
    graph = Graph(weights)
    problem = TSPProblem(graph, city_coords)
    
    # GA parameters
    population_size = 100  # Increased population size for more diversity
    max_iterations = 800   # Increased iterations for better convergence
    desired_fitness = 0    # We'll run for all iterations
    genetic_operator = TSPGeneticOperator(selection_prob=0.8, mutation_prob=0.1)
    
    # Create and run GA
    start_time = time.time()
    ga = GeneticAlgorithm(
        problem_interface=problem,
        genetic_operator=genetic_operator,
        population_size=population_size,
        max_iterations=max_iterations,
        desired_fitness=desired_fitness,
        verbosity=0  # Reduce default verbosity since we have custom logging
    )
    
    best_solution, best_fitness = ga.run()
    end_time = time.time()
    
    # Print results
    print("\nFinal Results:")
    print(f"Best Route: {' -> '.join(map(str, best_solution.representation))} -> {best_solution.representation[0]}")
    print(f"Total Distance: {best_fitness:.2f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Visualize results
    problem.plot_progress()
    problem.plot_route(best_solution)