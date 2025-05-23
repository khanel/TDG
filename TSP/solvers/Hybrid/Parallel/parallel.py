"""
Parallel Hybrid Approach

This module implements a parallel hybrid approach for the Traveling Salesperson Problem (TSP)
where multiple metaheuristic algorithms (GA, GWO, IGWO) run in parallel and share their
best solutions periodically.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from TSP.TSP import TSPProblem, Graph
from GA.GA import GeneticAlgorithm
from GWO.GWO import GrayWolfOptimization
from IGWO.IGWO import IGWO
from Core.problem import Solution
from TSP.solvers.GA.tsp_ga_solver import TSPGeneticOperator

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

def visualize_convergence(history, save_path=None):
    """
    Visualize the convergence history.
    
    Args:
        history: List of tuples (iteration, algorithm, fitness)
        save_path: Path to save the plot image (optional)
    """
    if not history:
        print("No history available to plot convergence")
        return
        
    plt.figure(figsize=(12, 6))
    
    # Group history by algorithm
    alg_data = {}
    for iteration, alg_name, fitness in history:
        if fitness is not None:
            if alg_name not in alg_data:
                alg_data[alg_name] = {'iterations': [], 'fitness': []}
            alg_data[alg_name]['iterations'].append(iteration)
            alg_data[alg_name]['fitness'].append(fitness)
    
    # Plot each algorithm's progress
    alg_colors = {'GA': 'green', 'GWO': 'blue', 'IGWO': 'purple', 'Combined': 'red'}
    
    for alg_name, data in alg_data.items():
        color = alg_colors.get(alg_name, 'black')
        plt.plot(data['iterations'], data['fitness'], 'o-', color=color, label=alg_name, alpha=0.7, markersize=4)
    
    plt.title('Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Distance)')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to '{save_path}'")

def run_hybrid_parallel(
    num_cities=20, 
    population_size=500, 
    max_iterations=2000, 
    sharing_interval=10,
    seed=42, 
    visualize=True,
    save_route_plot=True,
    save_convergence_plot=True,
    results_dir=None
):
    """
    Run the parallel hybrid approach on a TSP problem.
    
    Args:
        num_cities: Number of cities in the TSP problem
        population_size: Size of the population for each algorithm
        max_iterations: Maximum number of iterations
        sharing_interval: How often to share best solutions between algorithms
        seed: Random seed for reproducibility
        visualize: Whether to visualize the results
        save_route_plot: Whether to save the route plot
        save_convergence_plot: Whether to save the convergence plot
        results_dir: Directory to save results (default: current directory)
        
    Returns:
        A tuple containing (best_solution, elapsed_time, history)
    """
    # Use the Parallel directory for results if not specified
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
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
    
    # Initialize algorithms
    for alg in algorithms.values():
        alg.initialize()
    
    # Print optimization details
    print(f"Starting parallel optimization with {max_iterations} iterations...")
    print(f"Problem: TSP with {num_cities} cities")
    print(f"Algorithms: {', '.join(algorithms.keys())}")
    print(f"Population size: {population_size}")
    print(f"Sharing interval: {sharing_interval}")
    print("-" * 50)
    
    # Track history and best solution
    history = []
    best_solution = None
    best_fitness = float('inf')
    
    # Run optimization
    start_time = time.time()
    
    # Report initial state
    for alg_name, alg in algorithms.items():
        if alg.best_solution:
            fitness = alg.best_solution.fitness
            history.append((0, alg_name, fitness))
            print(f"Initial {alg_name} fitness: {fitness}")
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = alg.best_solution.copy()
                
    # Add combined best to history
    if best_solution:
        history.append((0, 'Combined', best_fitness))
    
    # Run algorithms in parallel with periodic sharing
    progress_interval = max(1, max_iterations // 10)  # Show progress every 10% of iterations
    
    for i in range(max_iterations):
        # Run one step of each algorithm
        for alg_name, alg in algorithms.items():
            alg.step()
            
            # Update history and best solution
            if alg.best_solution:
                fitness = alg.best_solution.fitness
                history.append((i+1, alg_name, fitness))
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = alg.best_solution.copy()
        
        # Add combined best to history
        history.append((i+1, 'Combined', best_fitness))
                
        # Share best solution among algorithms periodically
        if (i + 1) % sharing_interval == 0 and best_solution:
            for alg_name, alg in algorithms.items():
                # Replace the worst solution in each algorithm's population with the global best
                if alg.population:
                    alg.population.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
                    alg.population[-1] = best_solution.copy()
        
        # Progress reporting
        if (i + 1) % progress_interval == 0 or i == 0:
            print(f"Iteration {i+1}/{max_iterations} - Best fitness: {best_fitness}")
    
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
        route_path = os.path.join(results_dir, "route.png") if save_route_plot else None
        convergence_path = os.path.join(results_dir, "convergence.png") if save_convergence_plot else None
        
        visualize_results(best_solution, city_coords, route_path)
        visualize_convergence(history, convergence_path)
        
        # Show the plots
        plt.show()
    
    return best_solution, elapsed_time, history

if __name__ == "__main__":
    # Run the parallel hybrid approach with default parameters
    run_hybrid_parallel()
