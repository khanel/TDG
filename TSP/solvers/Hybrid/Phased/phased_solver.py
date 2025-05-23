"""
Phased Hybrid Solver Module

This module implements a phased hybrid optimization approach for TSP combining:
1. IGWO (Improved Grey Wolf Optimization) for exploration
2. GWO (Grey Wolf Optimization) for exploration
3. GA (Genetic Algorithm) for exploitation

The solver sequentially applies these algorithms, transferring the best solutions
between phases to improve overall performance.
"""
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any, Union

# Ensure the project root (TDG) is in PYTHONPATH for robust imports in __main__
# and for consistency if this script is moved or called differently.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from TSP.TSP import TSPProblem, Graph
from GA.GA import GeneticAlgorithm
from GWO.GWO import GrayWolfOptimization
from IGWO.IGWO import IGWO
from TSP.solvers.GA.tsp_ga_solver import TSPGeneticOperator
from Core.problem import Solution

class PhasedHybridSolver:
    """
    Implements a phased hybrid optimization approach for TSP problems.
    
    This solver sequentially applies IGWO (exploration), GWO (exploration), and GA
    (exploitation) algorithms, transferring the best solutions between phases.
    """
    
    def __init__(
        self,
        tsp_problem: TSPProblem,
        population_size: int,
        total_max_iterations: int,
        igwo_iteration_share: float = 0.3,
        gwo_iteration_share: float = 0.3,
        ga_mutation_rate: float = 0.1,
        ga_crossover_rate: float = 0.8,
        verbose: bool = True
    ):
        """
        Initialize the Phased Hybrid Solver.
        
        Args:
            tsp_problem: The TSPProblem instance.
            population_size: Population size for each algorithm.
            total_max_iterations: Total iterations for the entire hybrid process.
            igwo_iteration_share: Proportion of total iterations for IGWO (0.0 to 1.0).
            gwo_iteration_share: Proportion of total iterations for GWO (0.0 to 1.0).
                                GA will receive the remaining iterations.
            ga_mutation_rate: Mutation rate for the GA phase.
            ga_crossover_rate: Crossover rate for the GA phase.
            verbose: If True, prints progress and results.
        """
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.total_max_iterations = total_max_iterations
        self.igwo_iteration_share = igwo_iteration_share
        self.gwo_iteration_share = gwo_iteration_share
        self.ga_mutation_rate = ga_mutation_rate
        self.ga_crossover_rate = ga_crossover_rate
        self.verbose = verbose
        
        # Validate iteration shares
        if not (0 <= igwo_iteration_share <= 1 and 0 <= gwo_iteration_share <= 1):
            raise ValueError("Iteration shares must be between 0 and 1.")
        if igwo_iteration_share + gwo_iteration_share > 1:
            raise ValueError("Sum of IGWO and GWO iteration shares cannot exceed 1.")
        
        # Calculate iterations for each phase
        self.igwo_iters = int(total_max_iterations * igwo_iteration_share)
        self.gwo_iters = int(total_max_iterations * gwo_iteration_share)
        self.ga_iters = total_max_iterations - self.igwo_iters - self.gwo_iters
        
        # Initialize performance tracking variables
        self.best_solution = None
        self.best_fitness = float(np.inf)
        self.total_time = 0
        self.history = []  # To store the history of best fitness values

    def run(self) -> Tuple[Optional[List[int]], float, float]:
        """
        Run the phased hybrid optimization process.
        
        Returns:
            A tuple (best_solution, best_fitness, total_time).
        """
        start_time = time.time()
        
        # Reset history for this run
        self.history = []
        current_iteration = 0
        
        if self.verbose:
            print("Starting Phased Hybrid Solver...")
            print(f"Total Iterations: {self.total_max_iterations}")
            print(f"  IGWO (Exploration) Iterations: {self.igwo_iters}")
            print(f"  GWO (Exploration) Iterations: {self.gwo_iters}")
            print(f"  GA (Exploitation) Iterations: {self.ga_iters}")
        
        # Check for zero iterations edge case
        if self.igwo_iters == 0 and self.gwo_iters == 0 and self.ga_iters == 0 and self.total_max_iterations > 0:
            # If shares result in zero iterations for all, but total_max_iterations is positive,
            # it's likely due to very small shares and rounding. Give all to GA as a fallback.
            if self.verbose:
                print("Warning: Iteration shares resulted in zero iterations for all phases. Assigning all to GA.")
            self.ga_iters = self.total_max_iterations
        
        # --- Phase 1: IGWO (Exploration) ---
        if self.igwo_iters > 0:
            if self.verbose:
                print(f"\n--- Phase 1: IGWO (Exploration) for {self.igwo_iters} iterations ---")
            
            # Create and run IGWO solver
            igwo_solver = IGWO(
                problem=self.tsp_problem,
                population_size=self.population_size,
                max_iterations=self.igwo_iters
            )
            
            # Initialize and run through manual steps for better control
            igwo_solver.initialize()
            
            for i in range(self.igwo_iters):
                igwo_solver.step()
                current_iteration += 1
                
                # Record history after each step
                if igwo_solver.best_solution:
                    self.history.append({
                        'iteration': current_iteration,
                        'fitness': igwo_solver.best_solution.fitness,
                        'phase': 'IGWO'
                    })
                
            # Update best solution if improved
            if igwo_solver.best_solution and (self.best_solution is None or 
                                             igwo_solver.best_solution.fitness < self.best_fitness):
                self.best_solution = igwo_solver.best_solution
                self.best_fitness = igwo_solver.best_solution.fitness
            
            if self.verbose:
                igwo_best_fitness = igwo_solver.best_solution.fitness if igwo_solver.best_solution else float('inf')
                print(f"IGWO completed. Phase best fitness: {igwo_best_fitness}. Overall best: {self.best_fitness}")
        
        # --- Phase 2: GWO (Exploration) ---
        if self.gwo_iters > 0:
            if self.verbose:
                print(f"\n--- Phase 2: GWO (Exploration) for {self.gwo_iters} iterations ---")
            
            # Create and run GWO solver
            gwo_solver = GrayWolfOptimization(
                problem=self.tsp_problem,
                population_size=self.population_size,
                max_iterations=self.gwo_iters
            )
            
            # Initialize GWO
            gwo_solver.initialize()
            
            # If we have a solution from IGWO, seed it into the GWO population
            if self.best_solution:
                # Replace the worst solution in the population with our best
                if len(gwo_solver.population) > 0:
                    gwo_solver.population.sort(key=lambda x: x.fitness)
                    gwo_solver.population[-1] = self.best_solution.copy()
            
            # Run GWO solver
            for i in range(self.gwo_iters):
                gwo_solver.step()
                current_iteration += 1
                
                # Record history after each step
                if gwo_solver.best_solution:
                    self.history.append({
                        'iteration': current_iteration,
                        'fitness': gwo_solver.best_solution.fitness,
                        'phase': 'GWO'
                    })
            
            # Update best solution if improved
            if gwo_solver.best_solution and (self.best_solution is None or 
                                           gwo_solver.best_solution.fitness < self.best_fitness):
                self.best_solution = gwo_solver.best_solution
                self.best_fitness = gwo_solver.best_solution.fitness
            
            if self.verbose:
                gwo_best_fitness = gwo_solver.best_solution.fitness if gwo_solver.best_solution else float('inf')
                print(f"GWO completed. Phase best fitness: {gwo_best_fitness}. Overall best: {self.best_fitness}")
        
        # --- Phase 3: GA (Exploitation) ---
        if self.ga_iters > 0:
            if self.verbose:
                print(f"\n--- Phase 3: GA (Exploitation) for {self.ga_iters} iterations ---")
            
            # Create GA operator for TSP
            ga_operator = TSPGeneticOperator(
                mutation_prob=self.ga_mutation_rate,
                selection_prob=self.ga_crossover_rate
            )
            
            # Create and run GA solver
            ga_solver = GeneticAlgorithm(
                problem=self.tsp_problem,
                population_size=self.population_size,
                genetic_operator=ga_operator,
                max_iterations=self.ga_iters,
                mutation_rate=self.ga_mutation_rate,
                crossover_rate=self.ga_crossover_rate
            )
            
            # Initialize GA
            ga_solver.initialize()
            
            # If we have a best solution from previous phases, seed it into the GA population
            if self.best_solution:
                # Replace the worst solution in the population with our best
                if len(ga_solver.population) > 0:
                    ga_solver.population.sort(key=lambda x: x.fitness)
                    ga_solver.population[-1] = self.best_solution.copy()
            
            # Run GA solver
            for i in range(self.ga_iters):
                ga_solver.step()
                current_iteration += 1
                
                # Record history after each step
                if ga_solver.best_solution:
                    self.history.append({
                        'iteration': current_iteration,
                        'fitness': ga_solver.best_solution.fitness,
                        'phase': 'GA'
                    })
            
            # Update best solution if improved
            if ga_solver.best_solution and (self.best_solution is None or 
                                          ga_solver.best_solution.fitness < self.best_fitness):
                self.best_solution = ga_solver.best_solution
                self.best_fitness = ga_solver.best_solution.fitness
            
            if self.verbose:
                ga_best_fitness = ga_solver.best_solution.fitness if ga_solver.best_solution else float('inf')
                print(f"GA completed. Phase best fitness: {ga_best_fitness}. Overall best: {self.best_fitness}")
        
        # Fallback mechanism if no solution was found but iterations > 0
        if self.best_solution is None and self.total_max_iterations > 0:
            if self.verbose:
                print("Warning: No solution found. This might happen if all iteration counts were zero.")
            
            # Check if all phase iterations were 0 (shouldn't happen with earlier check, but just in case)
            if self.igwo_iters == 0 and self.gwo_iters == 0 and self.ga_iters == 0:
                if self.verbose:
                    print("Attempting a minimal GA run as a fallback.")
                
                # Create a minimal GA configuration
                fallback_ga_iters = max(1, int(self.total_max_iterations * 0.1))  # 10% or at least 1 iter
                ga_operator = TSPGeneticOperator(
                    mutation_prob=self.ga_mutation_rate, 
                    selection_prob=self.ga_crossover_rate
                )
                
                ga_solver = GeneticAlgorithm(
                    problem=self.tsp_problem,
                    population_size=self.population_size,
                    genetic_operator=ga_operator,
                    max_iterations=fallback_ga_iters
                )
                
                # Run the fallback GA
                ga_solver.initialize()
                for _ in range(fallback_ga_iters):
                    ga_solver.step()
                
                if ga_solver.best_solution:
                    self.best_solution = ga_solver.best_solution
                    self.best_fitness = ga_solver.best_solution.fitness
                    
                    if self.verbose:
                        print(f"Fallback GA completed. Fitness: {self.best_fitness}")
        
        # Calculate total runtime
        end_time = time.time()
        self.total_time = end_time - start_time
        
        if self.verbose:
            print(f"\nPhased Hybrid Solver finished in {self.total_time:.2f} seconds.")
            print(f"Final Overall Best Solution Fitness: {self.best_fitness}")
            # Uncomment to print full route: print(f"Final Overall Best Solution Path: {self.best_solution.representation}")
        
        # Return results
        best_route = self.best_solution.representation if self.best_solution else None
        return best_route, self.best_fitness, self.total_time

    def visualize_results(self, save_path=None):
        """
        Visualize the best route found.
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        plt.figure(figsize=(10, 10))
        coords = np.array(self.tsp_problem.city_coords)
        
        # Convert solution to route index format
        route = np.array(self.best_solution.representation + [self.best_solution.representation[0]]) - 1  # Add return to starting city
        
        # Plot the route
        plt.plot(coords[route, 0], coords[route, 1], 'o-', label='Best Route')
        
        # Add city labels
        for i, (x, y) in enumerate(coords, 1):
            plt.text(x, y, str(i), fontsize=12, ha='right')
        
        # Calculate route distance for the title
        total_distance = 0
        for i in range(len(route) - 1):
            city1, city2 = route[i], route[i + 1]
            total_distance += np.linalg.norm(coords[city1] - coords[city2])
        
        plt.title(f'Best TSP Route Found - Distance: {total_distance:.2f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Route plot saved to '{save_path}'")

    def visualize_convergence(self, save_path=None):
        """
        Visualize the convergence history.
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        if not self.history:
            print("No history available to plot convergence")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Extract data
        iterations = [entry['iteration'] for entry in self.history]
        fitness_values = [entry['fitness'] for entry in self.history]
        phases = [entry['phase'] for entry in self.history]
        
        # Plot fitness over iterations
        plt.plot(iterations, fitness_values, 'b-', alpha=0.5)
        plt.plot(iterations, fitness_values, 'ro', alpha=0.5, markersize=3)
        
        # Add markers for phase changes with different colors
        phase_colors = {'IGWO': 'purple', 'GWO': 'blue', 'GA': 'green'}
        
        # Plot points by phase
        for phase_name, color in phase_colors.items():
            phase_iterations = [entry['iteration'] for entry in self.history if entry['phase'] == phase_name]
            phase_fitness = [entry['fitness'] for entry in self.history if entry['phase'] == phase_name]
            if phase_iterations:
                plt.plot(phase_iterations, phase_fitness, 'o', color=color, markersize=5, label=phase_name)
        
        plt.title('Phased Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (Distance)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to '{save_path}'")

def run_phased_solver(
    tsp_problem: TSPProblem,
    population_size: int,
    total_max_iterations: int,
    igwo_iteration_share: float = 0.3,
    gwo_iteration_share: float = 0.3,
    ga_mutation_rate: float = 0.1,
    ga_crossover_rate: float = 0.8,
    verbose: bool = True
) -> Tuple[Optional[List[int]], float, float]:
    """
    Runs a phased hybrid optimization approach:
    1. IGWO (Exploration)
    2. GWO (Exploration)
    3. GA (Exploitation)

    This is a wrapper function that creates and runs a PhasedHybridSolver instance.

    Args:
        tsp_problem: The TSPProblem instance.
        population_size: Population size for each algorithm.
        total_max_iterations: Total iterations for the entire hybrid process.
        igwo_iteration_share: Proportion of total iterations for IGWO (0.0 to 1.0).
        gwo_iteration_share: Proportion of total iterations for GWO (0.0 to 1.0).
                                GA will receive the remaining iterations.
        ga_mutation_rate: Mutation rate for the GA phase.
        ga_crossover_rate: Crossover rate for the GA phase.
        verbose: If True, prints progress and results.

    Returns:
        A tuple (best_solution, best_fitness, total_time).
    """
    # Create and run the solver with the provided parameters
    solver = PhasedHybridSolver(
        tsp_problem=tsp_problem,
        population_size=population_size,
        total_max_iterations=total_max_iterations,
        igwo_iteration_share=igwo_iteration_share,
        gwo_iteration_share=gwo_iteration_share,
        ga_mutation_rate=ga_mutation_rate,
        ga_crossover_rate=ga_crossover_rate,
        verbose=verbose
    )
    
    return solver.run()

def run_hybrid_phased(
    num_cities=20,
    population_size=500,
    max_iterations=2000,
    seed=42,
    igwo_share=0.33,
    gwo_share=0.33,
    ga_mutation_rate=0.1,
    ga_crossover_rate=0.8,
    visualize=True,
    save_route_plot=True,
    save_convergence_plot=True,
    results_dir=None
):
    """
    Run the phased hybrid approach on a TSP problem.
    
    Args:
        num_cities: Number of cities in the TSP problem
        population_size: Size of the population for each algorithm
        max_iterations: Maximum number of iterations across all phases
        seed: Random seed for reproducibility
        igwo_share: Share of iterations for IGWO phase (0.0 to 1.0)
        gwo_share: Share of iterations for GWO phase (0.0 to 1.0)
        ga_mutation_rate: Mutation rate for the GA phase
        ga_crossover_rate: Crossover rate for the GA phase
        visualize: Whether to visualize the results
        save_route_plot: Whether to save the route plot
        save_convergence_plot: Whether to save the convergence plot
        results_dir: Directory to save results (default: current directory)
        
    Returns:
        A tuple containing (best_solution, best_fitness, elapsed_time)
    """
    # Use the Phased directory for results if not specified
    if results_dir is None:
        import os
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
    
    graph = Graph(weights=distances)
    tsp_problem = TSPProblem(graph, city_coords)
    
    # Validate iteration shares
    if igwo_share + gwo_share > 1.0:
        print("Warning: Sum of IGWO and GWO shares exceeds 1.0. Adjusting...")
        total = igwo_share + gwo_share
        igwo_share = igwo_share / total * 0.9  # Leave 10% for GA at minimum
        gwo_share = gwo_share / total * 0.9
        print(f"Adjusted shares: IGWO={igwo_share:.2f}, GWO={gwo_share:.2f}, GA={1-igwo_share-gwo_share:.2f}")
    
    # Create and configure the phased solver
    solver = PhasedHybridSolver(
        tsp_problem=tsp_problem,
        population_size=population_size,
        total_max_iterations=max_iterations,
        igwo_iteration_share=igwo_share,
        gwo_iteration_share=gwo_share,
        ga_mutation_rate=ga_mutation_rate,
        ga_crossover_rate=ga_crossover_rate,
        verbose=True
    )
    
    # Print optimization details
    print(f"Starting phased optimization with {max_iterations} iterations...")
    print(f"Problem: TSP with {num_cities} cities")
    print(f"Phase distribution:")
    print(f"  IGWO: {int(max_iterations * igwo_share)} iterations ({igwo_share*100:.1f}%)")
    print(f"  GWO: {int(max_iterations * gwo_share)} iterations ({gwo_share*100:.1f}%)")
    print(f"  GA: {int(max_iterations * (1-igwo_share-gwo_share))} iterations ({(1-igwo_share-gwo_share)*100:.1f}%)")
    print(f"Population size: {population_size}")
    print("-" * 50)
    
    # Run the phased optimization
    best_route, best_fitness, exec_time = solver.run()
    
    # Create a pseudo-solution object (for visualization purposes only)
    best_solution = None
    if best_route:
        best_solution = Solution(representation=best_route, problem=tsp_problem)
        best_solution.fitness = best_fitness
    
    # Print results
    print("\n" + "=" * 50)
    print("Phased optimization completed!")
    print(f"Time elapsed: {exec_time:.2f} seconds")
    print("\nBest solution found:")
    print(f"Fitness: {best_fitness}")
    print(f"Tour (first 10 cities): {best_route[:10]}...")
    print("=" * 50)
    
    # Visualize results if requested
    if visualize:
        import os
        route_path = os.path.join(results_dir, "route.png") if save_route_plot else None
        convergence_path = os.path.join(results_dir, "convergence.png") if save_convergence_plot else None
        
        # Use solver's visualization methods directly
        if save_route_plot and best_solution:
            solver.visualize_results(save_path=route_path)
            
        if save_convergence_plot and solver.history:
            solver.visualize_convergence(save_path=convergence_path)
        
        # Show the plots
        if visualize:
            plt.show()
    
    return best_solution, best_fitness, exec_time


if __name__ == "__main__":
    # This block allows direct execution for testing.
    # It uses the sys.path modification at the top of the file.
    
    print("Executing phased_solver.py as main script for testing.")

    # Sample graph data (adjust as needed, or load from a file)
    sample_graph_data = {
        0: (2, 2), 1: (2, 8), 2: (5, 5), 3: (6, 2),
        4: (6, 8), 5: (8, 5), 6: (10, 2), 7: (10, 8),
        8: (12, 4), 9: (12, 6) 
    }
    num_cities_example = len(sample_graph_data)
    
    graph_instance = Graph(num_nodes=num_cities_example, coordinates=sample_graph_data)
    tsp_problem_instance = TSPProblem(graph_instance)

    population_size_example = 50
    total_max_iterations_example = 300  # Total for the whole hybrid process
    
    igwo_share_example = 0.33  # ~33% for IGWO
    gwo_share_example = 0.33   # ~33% for GWO
    # GA gets the remaining ~34%

    print(f"\nRunning Phased Hybrid Solver with sample data:")
    print(f"Num cities: {num_cities_example}, Population: {population_size_example}, Total Iterations: {total_max_iterations_example}")
    print(f"IGWO share: {igwo_share_example*100:.1f}%, GWO share: {gwo_share_example*100:.1f}%")

    best_solution, best_fitness, exec_time = run_phased_solver(
        tsp_problem=tsp_problem_instance,
        population_size=population_size_example,
        total_max_iterations=total_max_iterations_example,
        igwo_iteration_share=igwo_share_example,
        gwo_iteration_share=gwo_share_example,
        ga_mutation_rate=0.05,
        ga_crossover_rate=0.7,
        verbose=True
    )

    print(f"\nExample Run Finished.")
    print(f"Execution Time: {exec_time:.2f}s")
    print(f"Best Fitness Found: {best_fitness}")
    if best_solution:
        print(f"Best Route (first 10 cities): {best_solution[:10]}...")
    else:
        print("No solution was found.")

    # Visualization (uncomment to save plots)
    # solver.visualize_results(save_path="best_route.png")
    # solver.visualize_convergence(save_path="convergence.png")

    run_hybrid_phased()