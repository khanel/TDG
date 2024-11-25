# Built-in Python packages
import sys
from pathlib import Path

# Third-party packages
import numpy as np
import matplotlib.pyplot as plt

# Local project modules
sys.path.append(str(Path(__file__).parent.parent))
from GA.Problem import ProblemInterface, Solution

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
        # If we received a Solution object, use its representation
        if isinstance(individual_representation, Solution):
            individual_representation = individual_representation.representation
        return self.calculate_path_distance(individual_representation)

    def validate_individual(self, individual):
        return len(individual.representation) == len(self.cities_graph.get_vertices()) and len(
            set(individual.representation)) == len(individual.representation)

    def get_individual_size(self):
        return len(self.cities_graph.get_vertices())

    def calculate_path_distance(self, path):
        # If we received a Solution object, use its representation
        if isinstance(path, Solution):
            path = path.representation
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


class Graph:
    def __init__(self, weights):
        """Initialize a graph with a distance matrix.
        
        Args:
            weights (numpy.ndarray): Square matrix of distances between cities
        """
        self.weights = weights
        self.vertices = [i + 1 for i in range(len(weights))]
        self.edges = []
        self.calculate_edges(weights)

    def calculate_edges(self, weights, first_index=0):
        """Calculate edges and their weights from the distance matrix.
        
        Args:
            weights (numpy.ndarray): Distance matrix
            first_index (int): Starting index for edge calculation
        
        Returns:
            list: List of tuples (city1, city2, distance)
        """
        for i in range(first_index, len(self.vertices)):
            for j in range(len(self.vertices)):
                if i != j and weights[i][j] != 0:
                    self.edges.append((i + 1, j + 1, weights[i][j]))
        return self.edges

    def add_edge(self, weight):
        """Add a new city with its distances to existing cities.
        
        Args:
            weight (list): Distances from the new city to all existing cities
        
        Returns:
            list: Updated list of edges
        """
        self.vertices.append(len(self.vertices) + 1)
        for i, j in enumerate(weight):
            if weight[i] != 0:
                self.edges.append((i + 1, len(self.vertices), weight[i]))
                self.edges.append((len(self.vertices), i + 1, weight[i]))
        return self.edges

    def get_weights(self):
        """Get the distance matrix.
        
        Returns:
            numpy.ndarray: Matrix of distances between cities
        """
        return self.weights

    def get_vertices(self):
        """Get the list of cities.
        
        Returns:
            list: List of city indices (1-based)
        """
        return self.vertices