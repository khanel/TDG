"""
Round-Robin Hybrid Approach

This module implements a round-robin hybrid approach for the Traveling Salesperson Problem (TSP)
using multiple metaheuristic algorithms (GA, GWO, IGWO) orchestrated in a round-robin fashion.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from TSP.TSP import TSPProblem, Graph
from GA.GA import GeneticAlgorithm
from GWO.GWO import GrayWolfOptimization
from IGWO.IGWO import IGWO
from Core.orchestrator import HybridSearchOrchestrator
from TSP.solvers.GA.tsp_ga_solver import TSPGeneticOperator

def round_robin_strategy(iteration, algs):
    """
    Simple round-robin strategy that cycles through algorithms.
    
    Args:
        iteration: Current iteration number
        algs: Dictionary of algorithm names to algorithm instances
    
    Returns:
        Name of the algorithm to use for this iteration
    """
    keys = list(algs.keys())
    return keys[iteration % len(keys)]

def setup_algorithms(tsp_problem, population_size, max_iterations):
    """
    Create and configure the algorithms used in the hybrid approach.
    
    Args:
        tsp_problem: The TSP problem instance
        population_size: Size of the population for each algorithm
        max_iterations: Maximum number of iterations
        
    Returns:
        Dictionary mapping algorithm names to configured algorithm instances
    """
    # Create a TSPGeneticOperator instance for the GA
    genetic_operator = TSPGeneticOperator(selection_prob=0.8, mutation_prob=0.1)

    # Initialize algorithms
    ga = GeneticAlgorithm(
        problem=tsp_problem,
        population_size=population_size,
        genetic_operator=genetic_operator,
        max_iterations=max_iterations
    )
    
    gwo = GrayWolfOptimization(
        problem=tsp_problem,
        population_size=population_size,
        max_iterations=max_iterations
    )
    
    igwo = IGWO(
        problem=tsp_problem,
        population_size=population_size,
        max_iterations=max_iterations
    )

    # Return algorithms dictionary
    return {
        'GA': ga,
        'GWO': gwo,
        'IGWO': igwo
    }

def visualize_results(best_solution, city_coords, save_path=None):
    """
    Visualize the best route found.
    
    Args:
        best_solution: The best solution found
        city_coords: Coordinates of the cities
        save_path: Path to save the plot image (optional)
    """
    plt.figure(figsize=(10, 10))
    coords = np.array(city_coords)
    route = np.array(best_solution.representation + [best_solution.representation[0]]) - 1
    plt.plot(coords[route, 0], coords[route, 1], 'o-', label='Best Route')
    
    for i, (x, y) in enumerate(coords, 1):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    
    plt.title(f'Best TSP Route Found - Distance: {best_solution.fitness:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Route plot saved to '{save_path}'")

def visualize_convergence(orchestrator, save_path=None):
    """
    Visualize the convergence history.
    
    Args:
        orchestrator: The HybridSearchOrchestrator instance
        save_path: Path to save the plot image (optional)
    """
    if not orchestrator.history:
        print("No history available to plot convergence")
        return
        
    plt.figure(figsize=(12, 6))
    iterations = [h[0] for h in orchestrator.history]
    fitness_values = [h[2] for h in orchestrator.history if h[2] is not None]
    iteration_points = [h[0] for h in orchestrator.history if h[2] is not None]
    
    if fitness_values:  # Only plot if we have valid fitness values
        plt.plot(iteration_points, fitness_values, 'b-', alpha=0.5)
        plt.plot(iteration_points, fitness_values, 'ro', alpha=0.5, markersize=3)
        
        # Add markers for algorithm changes
        alg_colors = {'GA': 'green', 'GWO': 'blue', 'IGWO': 'purple'}
        for i, h in enumerate(orchestrator.history):
            if h[2] is not None:  # Only plot if we have a valid fitness
                plt.plot(h[0], h[2], 'o', color=alg_colors.get(h[1], 'black'), markersize=5)
        
        # Add a legend for algorithms
        for alg_name, color in alg_colors.items():
            plt.plot([], [], 'o', color=color, label=alg_name)
            
        plt.title('Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (Distance)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to '{save_path}'")

def run_hybrid_round_robin(
    num_cities=20, 
    population_size=500, 
    max_iterations=2000, 
    seed=42, 
    visualize=True,
    save_route_plot=True,
    save_convergence_plot=True
):
    """
    Run the round-robin hybrid approach on a TSP problem.
    
    Args:
        num_cities: Number of cities in the TSP problem
        population_size: Size of the population for each algorithm
        max_iterations: Maximum number of iterations
        seed: Random seed for reproducibility
        visualize: Whether to visualize the results
        save_route_plot: Whether to save the route plot
        save_convergence_plot: Whether to save the convergence plot
        
    Returns:
        A tuple containing (best_solution, elapsed_time, orchestrator)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate random TSP instance
    city_coords = np.random.rand(num_cities, 2) * 100
    
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distances[i, j] = np.linalg.norm(city_coords[i] - city_coords[j])
    
    graph = Graph(distances)
    tsp_problem = TSPProblem(graph, city_coords)
    
    # Setup algorithms
    algorithms = setup_algorithms(tsp_problem, population_size, max_iterations)
    
    # Create orchestrator
    orchestrator = HybridSearchOrchestrator(
        problem=tsp_problem,
        algorithms=algorithms,
        strategy=round_robin_strategy,
        max_iterations=max_iterations
    )
    
    # Print optimization details
    print(f"Starting optimization with {max_iterations} iterations...")
    print(f"Problem: TSP with {num_cities} cities")
    print(f"Algorithms: {', '.join(algorithms.keys())}")
    print(f"Population size: {population_size}")
    print("-" * 50)
    
    # Initialize and run optimization
    start_time = time.time()
    orchestrator.initialize(population_size)
    
    print("Initial population created")
    print("First solution representation:", orchestrator.shared_population[0].representation)
    print("Initial best fitness:", orchestrator.best_solution.fitness if orchestrator.best_solution else "None")
    
    # Run with progress updates
    progress_interval = max(1, max_iterations // 10)  # Show progress every 10% of iterations
    for i in range(max_iterations):
        orchestrator.step()
        if (i + 1) % progress_interval == 0 or i == 0:
            current_best = orchestrator.best_solution.fitness if orchestrator.best_solution else "None"
            alg_name = orchestrator.history[-1][1] if orchestrator.history else "N/A"
            print(f"Iteration {i+1}/{max_iterations} - Best fitness: {current_best} - Algorithm: {alg_name}")
    
    best_solution = orchestrator.best_solution
    
    # Ensure the best solution has a fitness value
    if best_solution and best_solution.fitness is None:
        best_solution.evaluate()
        
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 50)
    print("Optimization completed!")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print("\nBest solution found:")
    print(best_solution)
    print("Tour:", best_solution.representation)
    print("Total distance:", best_solution.fitness)
    print("=" * 50)
    
    # Visualize results if requested
    if visualize:
        route_path = "tsp_best_route.png" if save_route_plot else None
        convergence_path = "tsp_convergence.png" if save_convergence_plot else None
        
        visualize_results(best_solution, city_coords, route_path)
        visualize_convergence(orchestrator, convergence_path)
        
        # Show the plots
        plt.show()
    
    return best_solution, elapsed_time, orchestrator

if __name__ == "__main__":
    # Run the round-robin hybrid approach with default parameters
    run_hybrid_round_robin()
