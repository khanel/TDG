"""
Phased Hybrid Solver Module

This module implements a phased hybrid optimization approach for TSP combining:
1. IGWO (Improved Grey Wolf Optimization) strictly for exploration
2. GWO (Grey Wolf Optimization) strictly for exploration
3. GA (Genetic Algorithm) strictly for exploitation

The solver sequentially applies these algorithms, transferring the entire population
of solutions between phases to improve overall performance while maintaining diversity.
Each algorithm is configured with parameters that emphasize their specific roles in the
exploration-exploitation spectrum.
"""
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any, Union
from tqdm import tqdm

# Ensure the project root (TDG) is in PYTHONPATH for robust imports in __main__
# and for consistency if this script is moved or called differently.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from problems.TSP.TSP import TSPProblem, Graph
from algorithms.GA.GA import GeneticAlgorithm
from algorithms.GWO.GWO import GrayWolfOptimization
from algorithms.IGWO.IGWO import IGWO
from problems.TSP.solvers.GA.tsp_ga_solver import TSPGeneticOperator
from Core.problem import Solution

# Import the new components
from problems.TSP.solvers.Hybrid.Phased.adaptive_parameters import AdaptiveParameterSystem
from problems.TSP.solvers.Hybrid.Phased.diversity_manager import DiversityManager
from problems.TSP.solvers.Hybrid.Phased.solution_cache import SolutionCache
from problems.TSP.solvers.Hybrid.Phased.local_search import _apply_3opt_improvement

class PhasedHybridSolver:
    """
    Implements a phased hybrid optimization approach for TSP problems.
    
    This solver sequentially applies IGWO (strictly exploration), GWO (strictly exploration), 
    and GA (strictly exploitation) algorithms, transferring the entire population of solutions 
    between phases to maintain search diversity and quality. Each algorithm is configured with 
    parameters that reinforce their specific role in the search process.
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
        use_adaptive_params: bool = True,
        use_diversity_management: bool = True,
        use_solution_caching: bool = True,
        use_advanced_local_search: bool = True,
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
            use_adaptive_params: Whether to use adaptive parameter adjustment.
            use_diversity_management: Whether to manage population diversity.
            use_solution_caching: Whether to cache solutions to avoid redundant evaluations.
            use_advanced_local_search: Whether to use advanced local search techniques.
            verbose: If True, prints progress and results.
        """
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.total_max_iterations = total_max_iterations
        self.igwo_iteration_share = igwo_iteration_share
        self.gwo_iteration_share = gwo_iteration_share
        self.ga_mutation_rate = ga_mutation_rate
        self.ga_crossover_rate = ga_crossover_rate
        self.use_adaptive_params = use_adaptive_params
        self.use_diversity_management = use_diversity_management
        self.use_solution_caching = use_solution_caching
        self.use_advanced_local_search = use_advanced_local_search
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
        
        # Initialize adaptive parameter system if enabled
        if self.use_adaptive_params:
            self.adaptive_params = AdaptiveParameterSystem(
                initial_parameters={
                    'a_initial': 2.5,  # IGWO parameter
                    'k': 0.4,          # IGWO parameter
                    'b1': 0.3,         # IGWO parameter
                    'b2': 0.3,         # IGWO parameter
                    'exploration_rate': 0.8,  # GWO parameter
                    'mutation_rate': self.ga_mutation_rate * 0.3,  # GA parameter
                    'crossover_rate': self.ga_crossover_rate * 1.5,  # GA parameter
                    'elitism_rate': 0.3  # GA parameter
                },
                min_values={
                    'a_initial': 1.0,
                    'k': 0.1,
                    'b1': 0.1,
                    'b2': 0.1,
                    'exploration_rate': 0.3,
                    'mutation_rate': 0.01,
                    'crossover_rate': 0.4,
                    'elitism_rate': 0.1
                },
                max_values={
                    'a_initial': 4.0,
                    'k': 0.8,
                    'b1': 0.7,
                    'b2': 0.7,
                    'exploration_rate': 1.0,
                    'mutation_rate': 0.5,
                    'crossover_rate': 1.0,
                    'elitism_rate': 0.5
                }
            )
        else:
            self.adaptive_params = None
            
        # Initialize diversity manager if enabled
        if self.use_diversity_management:
            self.diversity_manager = DiversityManager(
                diversity_threshold=0.3,
                intervention_rate=0.1
            )
        else:
            self.diversity_manager = None
            
        # Initialize solution cache if enabled
        if self.use_solution_caching:
            self.solution_cache = SolutionCache(
                max_cache_size=10000
            )
        else:
            self.solution_cache = None
            
        # Advanced local search capabilities
        self.apply_3opt = self.use_advanced_local_search

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
            
            # Get IGWO parameters (either adaptive or default)
            if self.use_adaptive_params:
                a_initial = self.adaptive_params.get_parameter('a_initial')
                k = self.adaptive_params.get_parameter('k')
                b1 = self.adaptive_params.get_parameter('b1')
                b2 = self.adaptive_params.get_parameter('b2')
            else:
                a_initial = 2.5  # Default: Higher initial control parameter for more exploration
                k = 0.4          # Default: Increased exploration coefficient
                b1 = 0.3         # Default: Reduced personal best influence for more exploration
                b2 = 0.3         # Default: Reduced social influence for more exploration
            
            # Create and run IGWO solver with exploration-focused parameters
            igwo_solver = IGWO(
                problem=self.tsp_problem,
                population_size=self.population_size,
                max_iterations=self.igwo_iters,
                a_initial=a_initial,
                k=k,
                b1=b1,
                b2=b2
            )
            
            # Initialize and run through manual steps for better control
            igwo_solver.initialize()
            
            # Initialize progress tracking
            fitness_history = []
            diversity_history = []
            
            # Use tqdm for progress tracking with improved description
            IGWO_progress_bar = tqdm(range(self.igwo_iters), 
                                    desc=f"IGWO Exploration (Starting...)", 
                                    unit="iter", 
                                    disable=not self.verbose, 
                                    ncols=100)
            
            for i in IGWO_progress_bar:
                # Before stepping, check if we should apply diversity intervention
                if self.use_diversity_management and i > 0 and i % 10 == 0:
                    # Measure diversity every 10 iterations
                    diversity = self.diversity_manager.measure_diversity(igwo_solver.population)
                    diversity_history.append(diversity)
                    
                    # Apply diversity intervention if needed
                    if self.diversity_manager.should_intervene(igwo_solver.population):
                        igwo_solver.population = self.diversity_manager.apply_diversity_intervention(
                            igwo_solver.population,
                            self.tsp_problem,
                            igwo_solver.best_solution
                        )
                
                # Run one IGWO step
                igwo_solver.step()
                current_iteration += 1
                
                # Update progress bar with current best fitness
                if igwo_solver.best_solution:
                    IGWO_progress_bar.set_description(
                        f"IGWO Exploration (Best: {igwo_solver.best_solution.fitness:.2f})"
                    )
                    fitness_history.append(igwo_solver.best_solution.fitness)
                    
                    # Update adaptive parameters if enabled
                    if self.use_adaptive_params and i > 0 and i % 5 == 0:
                        self.adaptive_params.update_parameters(
                            fitness_history,
                            i,
                            'IGWO'
                        )
                        
                        # Get updated parameters
                        a_initial = self.adaptive_params.get_parameter('a_initial')
                        k = self.adaptive_params.get_parameter('k')
                        b1 = self.adaptive_params.get_parameter('b1')
                        b2 = self.adaptive_params.get_parameter('b2')
                        
                        # Update IGWO parameters
                        igwo_solver.a_initial = a_initial
                        igwo_solver.k = k
                        igwo_solver.b1 = b1
                        igwo_solver.b2 = b2
                
                # Record history after each step
                if igwo_solver.best_solution:
                    # Only record if the solution is better than all previous
                    current_fitness = igwo_solver.best_solution.fitness
                    if not self.history or current_fitness <= min(entry['fitness'] for entry in self.history):
                        self.history.append({
                            'iteration': current_iteration,
                            'fitness': current_fitness,
                            'phase': 'IGWO'
                        })
                    # Record periodic progress regardless
                    elif i % 10 == 0:
                        self.history.append({
                            'iteration': current_iteration,
                            'fitness': current_fitness,
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
                
                if self.use_diversity_management and diversity_history:
                    avg_diversity = sum(diversity_history) / len(diversity_history)
                    print(f"Average population diversity during IGWO: {avg_diversity:.4f}")
                    
                if self.use_adaptive_params:
                    print(f"Final IGWO parameters: a_initial={a_initial:.2f}, k={k:.2f}, b1={b1:.2f}, b2={b2:.2f}")
        
        # --- Phase 2: GWO (Exploration) ---
        if self.gwo_iters > 0:
            if self.verbose:
                print(f"\n--- Phase 2: GWO (Exploration) for {self.gwo_iters} iterations ---")
            
            # Get GWO parameters (either adaptive or default)
            if self.use_adaptive_params:
                exploration_rate = self.adaptive_params.get_parameter('exploration_rate')
            else:
                exploration_rate = 0.8  # Default: Slightly reduced exploration rate for better balance
            
            # Create and run GWO solver with exploration-focused parameters
            gwo_solver = GrayWolfOptimization(
                problem=self.tsp_problem,
                population_size=self.population_size,
                max_iterations=self.gwo_iters,
                exploration_rate=exploration_rate
            )
            
            # Initialize GWO
            gwo_solver.initialize()
            
            # Transfer solutions from IGWO to GWO population with enhanced transition
            if self.igwo_iters > 0 and igwo_solver and len(igwo_solver.population) > 0:
                # Apply phase transition optimization
                gwo_solver.population = self._prepare_phase_transition(
                    from_phase='IGWO',
                    to_phase='GWO',
                    population=igwo_solver.population,
                    best_solution=self.best_solution
                )
                
                if self.verbose:
                    print(f"Enhanced transition from IGWO to GWO phase complete")
            
            # Initialize progress tracking
            fitness_history = []
            diversity_history = []
            
            # Use tqdm for progress tracking with improved description
            GWO_progress_bar = tqdm(range(self.gwo_iters), 
                                   desc=f"GWO Exploration (Best: {self.best_fitness:.2f})", 
                                   unit="iter", 
                                   disable=not self.verbose, 
                                   ncols=100)
            
            for i in GWO_progress_bar:
                # Before stepping, check if we should apply diversity intervention
                if self.use_diversity_management and i > 0 and i % 10 == 0:
                    # Measure diversity every 10 iterations
                    diversity = self.diversity_manager.measure_diversity(gwo_solver.population)
                    diversity_history.append(diversity)
                    
                    # Apply diversity intervention if needed
                    if self.diversity_manager.should_intervene(gwo_solver.population):
                        gwo_solver.population = self.diversity_manager.apply_diversity_intervention(
                            gwo_solver.population,
                            self.tsp_problem,
                            gwo_solver.best_solution
                        )
                
                # Run one GWO step
                gwo_solver.step()
                current_iteration += 1
                
                # Update progress bar with current best fitness
                if gwo_solver.best_solution:
                    GWO_progress_bar.set_description(
                        f"GWO Exploration (Best: {gwo_solver.best_solution.fitness:.2f})"
                    )
                    fitness_history.append(gwo_solver.best_solution.fitness)
                    
                    # Update adaptive parameters if enabled
                    if self.use_adaptive_params and i > 0 and i % 5 == 0:
                        self.adaptive_params.update_parameters(
                            fitness_history,
                            i,
                            'GWO'
                        )
                        
                        # Get updated parameters
                        exploration_rate = self.adaptive_params.get_parameter('exploration_rate')
                        
                        # Update GWO parameters
                        gwo_solver.exploration_rate = exploration_rate
                
                # Record history after each step
                if gwo_solver.best_solution:
                    # Only record if the solution is better than all previous
                    current_fitness = gwo_solver.best_solution.fitness
                    if not self.history or current_fitness <= min(entry['fitness'] for entry in self.history):
                        self.history.append({
                            'iteration': current_iteration,
                            'fitness': current_fitness,
                            'phase': 'GWO'
                        })
                    # Record periodic progress regardless
                    elif i % 10 == 0:
                        self.history.append({
                            'iteration': current_iteration,
                            'fitness': current_fitness,
                            'phase': 'GWO'
                        })
                        
                # Apply 2-opt local search occasionally to best solutions for better convergence
                if i > 0 and i % 20 == 0:
                    # Apply to the best solution
                    if gwo_solver.best_solution:
                        improved_sol = self._apply_2opt_improvement(gwo_solver.best_solution)
                        if improved_sol and improved_sol.fitness < gwo_solver.best_solution.fitness:
                            gwo_solver.best_solution = improved_sol
                            gwo_solver.population[0] = improved_sol.copy()
            
            # Update best solution if improved
            if gwo_solver.best_solution and (self.best_solution is None or 
                                           gwo_solver.best_solution.fitness < self.best_fitness):
                self.best_solution = gwo_solver.best_solution
                self.best_fitness = gwo_solver.best_solution.fitness
            
            if self.verbose:
                gwo_best_fitness = gwo_solver.best_solution.fitness if gwo_solver.best_solution else float('inf')
                print(f"GWO completed. Phase best fitness: {gwo_best_fitness}. Overall best: {self.best_fitness}")
                
                if self.use_diversity_management and diversity_history:
                    avg_diversity = sum(diversity_history) / len(diversity_history)
                    print(f"Average population diversity during GWO: {avg_diversity:.4f}")
                    
                if self.use_adaptive_params:
                    print(f"Final GWO parameters: exploration_rate={exploration_rate:.2f}")
        
        # --- Phase 3: GA (Exploitation) ---
        if self.ga_iters > 0:
            if self.verbose:
                print(f"\n--- Phase 3: GA (Exploitation) for {self.ga_iters} iterations ---")
            
            # Get GA parameters (either adaptive or default)
            if self.use_adaptive_params:
                mutation_rate = self.adaptive_params.get_parameter('mutation_rate')
                crossover_rate = self.adaptive_params.get_parameter('crossover_rate')
                elitism_rate = self.adaptive_params.get_parameter('elitism_rate')
            else:
                mutation_rate = self.ga_mutation_rate * 0.3  # Default: Low mutation rate but not too restrictive
                crossover_rate = self.ga_crossover_rate * 1.5  # Default: Higher selection pressure for exploitation
                elitism_rate = 0.3  # Default: High elitism to preserve best solutions but allow some diversity
            
            # Create GA operator for TSP optimized for balanced exploitation
            ga_operator = TSPGeneticOperator(
                mutation_prob=mutation_rate,
                selection_prob=crossover_rate
            )
            
            # Create and run GA solver with balanced exploitation parameters
            ga_solver = GeneticAlgorithm(
                problem=self.tsp_problem,
                population_size=self.population_size,
                genetic_operator=ga_operator,
                max_iterations=self.ga_iters,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elitism_rate=elitism_rate
            )
            
            # Initialize GA
            ga_solver.initialize()
            
            # Transfer solutions from previous phase to GA population with enhanced transition
            if self.gwo_iters > 0 and gwo_solver and len(gwo_solver.population) > 0:
                # Apply phase transition optimization
                ga_solver.population = self._prepare_phase_transition(
                    from_phase='GWO',
                    to_phase='GA',
                    population=gwo_solver.population,
                    best_solution=self.best_solution
                )
                
                if self.verbose:
                    print(f"Enhanced transition from GWO to GA phase complete")
            # If no GWO phase was run but we have IGWO solutions, transfer those directly
            elif self.igwo_iters > 0 and igwo_solver and len(igwo_solver.population) > 0:
                # Apply phase transition optimization
                ga_solver.population = self._prepare_phase_transition(
                    from_phase='IGWO',
                    to_phase='GA',
                    population=igwo_solver.population,
                    best_solution=self.best_solution
                )
                
                if self.verbose:
                    print(f"Enhanced transition from IGWO to GA phase complete")
                
            # Initialize progress tracking
            fitness_history = []
            diversity_history = []
                
            # Run GA solver with balanced exploitation focus
            GA_progress_bar = tqdm(range(self.ga_iters), 
                                  desc=f"GA Exploitation (Best: {self.best_fitness:.2f})", 
                                  unit="iter", 
                                  disable=not self.verbose, 
                                  ncols=100)
            
            # Store best solution from previous phases to ensure we never get worse
            best_solution_from_phases = self.best_solution.copy() if self.best_solution else None
            
            # Make sure GA's best solution is set to our current best
            if best_solution_from_phases:
                ga_solver.best_solution = best_solution_from_phases.copy()
                ga_solver.best_fitness = best_solution_from_phases.fitness
                
            # Increase population diversity by introducing small mutations to some solutions
            # This helps GA escape local optima while still focusing on exploitation
            if len(ga_solver.population) > 5:  # Only if we have enough solutions
                for i in range(len(ga_solver.population) // 4):  # Mutate 25% of the population
                    # Skip the best solutions (kept in first positions by sorting)
                    idx = len(ga_solver.population) // 2 + i
                    if idx < len(ga_solver.population):
                        # Create a slightly mutated version of a good solution
                        base_sol_idx = np.random.randint(0, len(ga_solver.population) // 4)
                        base_sol = ga_solver.population[base_sol_idx].copy()
                        
                        # Small 2-opt local improvement
                        n = len(base_sol.representation)
                        for _ in range(2):  # Apply 2 small changes
                            # Choose two positions in the route (avoiding position 0 which is city 1)
                            pos1, pos2 = 1 + np.random.randint(0, n-1), 1 + np.random.randint(0, n-1)
                            if pos1 != pos2:
                                # Swap cities
                                base_sol.representation[pos1], base_sol.representation[pos2] = \
                                    base_sol.representation[pos2], base_sol.representation[pos1]
                        
                        # Re-evaluate and add to population
                        base_sol.evaluate()
                        ga_solver.population[idx] = base_sol
            
            # Main GA loop
            for i in GA_progress_bar:
                # Before stepping, check if we should apply diversity intervention
                if self.use_diversity_management and i > 0 and i % 15 == 0:
                    # Measure diversity every 15 iterations (less frequent for GA)
                    diversity = self.diversity_manager.measure_diversity(ga_solver.population)
                    diversity_history.append(diversity)
                    
                    # Apply diversity intervention if needed (limited for exploitation phase)
                    if self.diversity_manager.should_intervene(ga_solver.population) and diversity < 0.2:
                        # Only intervene if diversity is very low in exploitation phase
                        ga_solver.population = self.diversity_manager.apply_diversity_intervention(
                            ga_solver.population,
                            self.tsp_problem,
                            ga_solver.best_solution
                        )
                
                # Store the current best solution before stepping
                prev_best_sol = None
                if ga_solver.best_solution:
                    prev_best_sol = ga_solver.best_solution.copy()
                    pre_step_best_fitness = ga_solver.best_solution.fitness
                
                # Run one GA step
                ga_solver.step()
                current_iteration += 1
                
                # Update progress bar with current best fitness
                if ga_solver.best_solution:
                    GA_progress_bar.set_description(
                        f"GA Exploitation (Best: {ga_solver.best_solution.fitness:.2f})"
                    )
                    fitness_history.append(ga_solver.best_solution.fitness)
                
                # If the new solution is worse than previous best, revert to the previous best
                # but with a small chance to accept slightly worse solutions temporarily
                accept_worse = np.random.random() < 0.05  # 5% chance to accept worse solution
                
                if prev_best_sol and (ga_solver.best_solution is None or 
                                    (ga_solver.best_solution.fitness > prev_best_sol.fitness and not accept_worse)):
                    ga_solver.best_solution = prev_best_sol
                    
                    # Ensure the best solution is preserved
                    ga_solver.population[0] = prev_best_sol.copy()  # Replace the first solution
                    
                    # Every 10 iterations, apply local improvement to the best solution
                    if i % 10 == 0:
                        improved_solution = self._apply_2opt_improvement(prev_best_sol)
                        if improved_solution and improved_solution.fitness < prev_best_sol.fitness:
                            ga_solver.best_solution = improved_solution
                            ga_solver.population[0] = improved_solution.copy()
                            # Also insert at position 1 and at the end
                            ga_solver.population[1] = improved_solution.copy()
                            ga_solver.population[-1] = improved_solution.copy()
                
                # Update overall best solution if GA found a better one
                if ga_solver.best_solution and (self.best_solution is None or 
                                             ga_solver.best_solution.fitness < self.best_fitness):
                    self.best_solution = ga_solver.best_solution.copy()
                    self.best_fitness = ga_solver.best_solution.fitness
                    
                    # Update progress bar with new best fitness
                    GA_progress_bar.set_description(f"GA Exploitation (Best: {self.best_fitness:.2f})")
                    
                    # Log improvement in history - ensure the GA phase is visible in the plot
                    self.history.append({
                        'iteration': current_iteration,
                        'fitness': self.best_fitness,
                        'phase': 'GA'  
                    })
                    
                    # Apply appropriate local search to the best solution
                    if self.use_advanced_local_search:
                        # Try 3-opt first for potentially larger improvements
                        improved_solution = _apply_3opt_improvement(self, self.best_solution)
                        if improved_solution and improved_solution.fitness < self.best_fitness:
                            self.best_solution = improved_solution.copy()
                            self.best_fitness = improved_solution.fitness
                            
                            # Update progress bar with the further improved fitness
                            GA_progress_bar.set_description(f"GA Exploitation (Best: {self.best_fitness:.2f})")
                            
                            # Log the additional improvement from 3-opt
                            self.history.append({
                                'iteration': current_iteration + 0.5,  # offset to show in plot
                                'fitness': self.best_fitness,
                                'phase': 'GA'
                            })
                    else:
                        # Apply regular 2-opt improvement
                        improved_solution = self._apply_2opt_improvement(self.best_solution)
                        if improved_solution and improved_solution.fitness < self.best_fitness:
                            self.best_solution = improved_solution.copy()
                            self.best_fitness = improved_solution.fitness
                            
                            # Update progress bar with the further improved fitness
                            GA_progress_bar.set_description(f"GA Exploitation (Best: {self.best_fitness:.2f})")
                            
                            # Log the additional improvement from 2-opt
                            self.history.append({
                                'iteration': current_iteration + 0.5,  # offset to show in plot
                                'fitness': self.best_fitness,
                                'phase': 'GA'
                            })
                
                # Record periodic progress for visualization purposes
                elif i % 10 == 0:
                    self.history.append({
                        'iteration': current_iteration,
                        'fitness': self.best_fitness,  # Use global best, not current iteration
                        'phase': 'GA'
                    })
                    
                    # Update adaptive parameters if enabled
                    if self.use_adaptive_params and i > 0 and i % 5 == 0:
                        self.adaptive_params.update_parameters(
                            fitness_history,
                            i,
                            'GA'
                        )
                        
                        # Get updated parameters
                        mutation_rate = self.adaptive_params.get_parameter('mutation_rate')
                        crossover_rate = self.adaptive_params.get_parameter('crossover_rate')
                        elitism_rate = self.adaptive_params.get_parameter('elitism_rate')
                        
                        # Update GA parameters
                        ga_solver.mutation_rate = mutation_rate
                        ga_solver.crossover_rate = crossover_rate
                        ga_solver.elitism_rate = elitism_rate
                        ga_operator.mutation_prob = mutation_rate
                        ga_operator.selection_prob = crossover_rate
                    
                    # Every 30 iterations, try more aggressive local search on the best solution
                    if i % 30 == 0:  # Less frequently for more intensive search
                        # Try multiple improvements in sequence
                        current_best = self.best_solution.copy()
                        
                        if self.use_advanced_local_search:
                            # Try both 2-opt and 3-opt in sequence
                            for _ in range(2):  # Multiple attempts
                                # Try 2-opt first (faster)
                                improved = self._apply_2opt_improvement(current_best)
                                if improved and improved.fitness < current_best.fitness:
                                    current_best = improved
                                
                                # Then try 3-opt (more powerful but slower)
                                improved = _apply_3opt_improvement(self, current_best)
                                if improved and improved.fitness < current_best.fitness:
                                    current_best = improved
                        else:
                            # Just use multiple 2-opt improvements
                            for _ in range(3):  # Try multiple improvement attempts
                                improved = self._apply_2opt_improvement(current_best)
                                if improved and improved.fitness < current_best.fitness:
                                    current_best = improved
                        
                        # If we found a better solution, update everything
                        if current_best.fitness < self.best_fitness:
                            self.best_solution = current_best
                            self.best_fitness = current_best.fitness
                            ga_solver.best_solution = current_best.copy()
                            
                            # Add improved solution to population
                            ga_solver.population[0] = current_best.copy()
                            ga_solver.population[-1] = current_best.copy()
                            
                            # Update progress bar
                            GA_progress_bar.set_description(f"GA Exploitation (Best: {self.best_fitness:.2f})")
                            
                            # Log the deeper improvement in history
                            self.history.append({
                                'iteration': current_iteration,
                                'fitness': self.best_fitness,
                                'phase': 'GA'
                            })
            
            # After GA phase is complete, ensure we're using the absolute best solution
            if ga_solver.best_solution and ga_solver.best_solution.fitness < self.best_fitness:
                self.best_solution = ga_solver.best_solution.copy()
                self.best_fitness = ga_solver.best_solution.fitness
            
            if self.verbose:
                ga_best_fitness = ga_solver.best_solution.fitness if ga_solver.best_solution else float('inf')
                print(f"GA completed. Phase best fitness: {ga_best_fitness}. Overall best: {self.best_fitness}")
                
                if self.use_diversity_management and diversity_history:
                    avg_diversity = sum(diversity_history) / len(diversity_history)
                    print(f"Average population diversity during GA: {avg_diversity:.4f}")
                    
                if self.use_adaptive_params:
                    print(f"Final GA parameters: mutation_rate={mutation_rate:.2f}, crossover_rate={crossover_rate:.2f}, elitism_rate={elitism_rate:.2f}")
                    
                if self.use_solution_caching:
                    hit_rate = self.solution_cache.get_hit_rate()
                    cache_size = self.solution_cache.get_cache_size()
                    print(f"Solution cache: size={cache_size}, hit rate={hit_rate:.1f}%")
        
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
                    mutation_prob=self.ga_mutation_rate * 0.2,  # Very low mutation for strict exploitation
                    selection_prob=self.ga_crossover_rate * 1.5  # Much higher selection pressure for exploitation
                )
                
                ga_solver = GeneticAlgorithm(
                    problem=self.tsp_problem,
                    population_size=self.population_size,
                    genetic_operator=ga_operator,
                    max_iterations=fallback_ga_iters,
                    elitism_rate=0.3  # Higher elitism for strict exploitation
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

    def _apply_2opt_improvement(self, solution):
        """
        Apply a 2-opt local improvement to a TSP solution.
        This is a standard local search method for TSP that removes route crossings.
        
        Args:
            solution (Solution): The TSP solution to improve
            
        Returns:
            Solution: Improved solution, or None if no improvement was found
        """
        if not solution:
            return None
            
        # Create a copy to avoid modifying the original
        best_sol = solution.copy()
        best_fitness = best_sol.fitness
        
        # Get the route
        route = best_sol.representation.copy()
        n = len(route)
        
        improved = False
        
        # Get city coordinates for faster distance calculations
        coords = self.tsp_problem.city_coords
        
        # Try a more comprehensive set of 2-opt moves
        max_attempts = min(n * 3, 100)  # Increased number of attempts
        
        # First, try a systematic approach
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                # Calculate the change in tour length if we reverse the segment
                city1 = route[i-1] - 1  # Convert from 1-indexed to 0-indexed
                city2 = route[i] - 1
                city3 = route[j] - 1
                city4 = route[j+1] - 1
                
                # Calculate distances for current route segment
                if isinstance(coords[city1], tuple) and isinstance(coords[city2], tuple):
                    # Handle coordinates as tuples
                    d1 = np.sqrt((coords[city1][0] - coords[city2][0])**2 + 
                                 (coords[city1][1] - coords[city2][1])**2)
                    d2 = np.sqrt((coords[city3][0] - coords[city4][0])**2 + 
                                 (coords[city3][1] - coords[city4][1])**2)
                    
                    # Calculate distances for proposed route segment
                    d3 = np.sqrt((coords[city1][0] - coords[city3][0])**2 + 
                                 (coords[city1][1] - coords[city3][1])**2)
                    d4 = np.sqrt((coords[city2][0] - coords[city4][0])**2 + 
                                 (coords[city2][1] - coords[city4][1])**2)
                else:
                    # Handle coordinates as numpy arrays if applicable
                    d1 = np.linalg.norm(coords[city1] - coords[city2])
                    d2 = np.linalg.norm(coords[city3] - coords[city4])
                    
                    # Calculate distances for proposed route segment
                    d3 = np.linalg.norm(coords[city1] - coords[city3])
                    d4 = np.linalg.norm(coords[city2] - coords[city4])
                
                # If the new route would be shorter, apply the 2-opt swap
                if d1 + d2 > d3 + d4:
                    new_route = route.copy()
                    new_route[i:j+1] = route[i:j+1][::-1]
                    
                    # Create and evaluate new solution
                    new_sol = Solution(new_route, solution.problem)
                    new_sol.evaluate()
                    
                    # If improved, keep it
                    if new_sol.fitness < best_fitness:
                        best_sol = new_sol
                        best_fitness = new_sol.fitness
                        route = new_route
                        improved = True
                        break  # We found an improvement, break and continue with the new route
            
            # If we found an improvement, break out of the outer loop too
            if improved:
                break
                
        # If no improvement was found with systematic search, try random moves
        if not improved:
            for _ in range(max_attempts):
                # Select two random positions (avoiding the first city)
                i = 1 + np.random.randint(0, n-3)
                j = i + 1 + np.random.randint(0, n-i-1)
                
                # Apply 2-opt: reverse the segment between i and j
                new_route = route.copy()
                new_route[i:j+1] = route[i:j+1][::-1]
                
                # Create and evaluate new solution
                new_sol = Solution(new_route, solution.problem)
                new_sol.evaluate()
                
                # If improved, keep it
                if new_sol.fitness < best_fitness:
                    best_sol = new_sol
                    best_fitness = new_sol.fitness
                    route = new_route
                    improved = True
        
        return best_sol if improved else None
    
    def _prepare_phase_transition(self, from_phase, to_phase, population, best_solution):
        """
        Prepare for a transition between optimization phases by enhancing the population
        and applying specialized transformations.
        
        Args:
            from_phase: The phase transitioning from ('IGWO', 'GWO', or None).
            to_phase: The phase transitioning to ('GWO', 'GA', or None).
            population: The current population of solutions.
            best_solution: The current best solution.
            
        Returns:
            Enhanced population ready for the next phase.
        """
        if self.verbose:
            print(f"\nPreparing transition from {from_phase if from_phase else 'None'} to {to_phase}...")
        
        # Create a copy of the population to avoid modifying the original
        enhanced_population = [sol.copy() for sol in population]
        
        # Sort population by fitness
        enhanced_population.sort(key=lambda x: x.fitness)
        
        # Special handling based on the transition type
        if from_phase == 'IGWO' and to_phase == 'GWO':
            # IGWO to GWO: Enhance exploration capabilities
            if self.verbose:
                print("Applying IGWO to GWO transition enhancements...")
            
            # Increase diversity through targeted mutations
            for i in range(len(enhanced_population) // 4, len(enhanced_population) // 2):
                # Apply a more aggressive mutation to mid-quality solutions
                route = enhanced_population[i].representation.copy()
                n = len(route)
                
                # Apply three random swaps (avoiding position 0)
                for _ in range(3):
                    pos1, pos2 = 1 + np.random.randint(0, n-1), 1 + np.random.randint(0, n-1)
                    if pos1 != pos2:
                        route[pos1], route[pos2] = route[pos2], route[pos1]
                
                # Create and evaluate new solution
                new_sol = Solution(route, enhanced_population[i].problem)
                new_sol.evaluate()
                enhanced_population[i] = new_sol
        
        elif (from_phase == 'IGWO' or from_phase == 'GWO') and to_phase == 'GA':
            # Transition to GA: Prepare for exploitation phase
            if self.verbose:
                print("Applying transition enhancements for GA phase...")
            
            # Apply local search to the best solutions to create strong starting points for exploitation
            local_search_count = min(len(enhanced_population) // 10, 5)  # Up to 5 or 10% of population
            
            for i in range(local_search_count):
                # Apply 2-opt improvement to top solutions
                improved_sol = self._apply_2opt_improvement(enhanced_population[i])
                if improved_sol:
                    enhanced_population[i] = improved_sol
            
            # Create some solutions with good diversity characteristics
            if len(enhanced_population) >= 10:
                for i in range(len(enhanced_population) // 2, len(enhanced_population) // 2 + 3):
                    # Take a good solution and apply a diversity-inducing operation
                    base_sol = enhanced_population[i % 5].copy()  # Use one of the top 5 solutions
                    
                    # Apply insertion mutation: remove a city and insert it elsewhere
                    route = base_sol.representation.copy()
                    n = len(route)
                    
                    # Select a random segment (avoiding city 1 at position 0)
                    start = 1 + np.random.randint(0, n-3)
                    end = start + np.random.randint(1, min(n-start, 5))  # Segment length 1-5
                    
                    # Remove segment and insert elsewhere
                    segment = route[start:end]
                    remaining = route[:start] + route[end:]
                    
                    # Choose insertion point
                    insert_pos = 1 + np.random.randint(0, len(remaining)-1)
                    
                    # Create new route
                    new_route = remaining[:insert_pos] + segment + remaining[insert_pos:]
                    
                    # Create and evaluate new solution
                    new_sol = Solution(new_route, base_sol.problem)
                    new_sol.evaluate()
                    
                    # Only use if reasonably good
                    if new_sol.fitness < 1.5 * enhanced_population[0].fitness:
                        enhanced_population[i] = new_sol
        
        # Ensure the best solution is always preserved
        if best_solution:
            # Make sure the best solution is in the first position
            enhanced_population[0] = best_solution.copy()
            
            # Add best solution at another position for additional safety
            enhanced_population[-1] = best_solution.copy()
        
        # Ensure all solutions are evaluated
        for sol in enhanced_population:
            if sol.fitness is None:
                sol.evaluate()
        
        if self.verbose:
            print(f"Population enhanced for {to_phase} phase.")
        
        return enhanced_population
    
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
        Visualize the convergence history with enhanced plotting features.
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        if not self.history:
            print("No history available to plot convergence")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Extract data
        iterations = [entry['iteration'] for entry in self.history]
        fitness_values = [entry['fitness'] for entry in self.history]
        phases = [entry['phase'] for entry in self.history]
        
        # Calculate best fitness at each iteration (monotonically decreasing)
        best_fitness = []
        current_best = float('inf')
        for fitness in fitness_values:
            if fitness < current_best:
                current_best = fitness
            best_fitness.append(current_best)
        
        # Plot raw fitness data (lighter, shows actual algorithm behavior)
        plt.plot(iterations, fitness_values, 'b-', alpha=0.2, linewidth=1, label='Raw Fitness')
        plt.plot(iterations, fitness_values, 'ro', alpha=0.15, markersize=3)
        
        # Plot the monotonically decreasing best fitness (thicker line)
        plt.plot(iterations, best_fitness, 'k-', linewidth=2.5, label='Best Overall')
        
        # Add markers for phase changes with different colors
        phase_colors = {'IGWO': 'purple', 'GWO': 'blue', 'GA': 'green'}
        
        # Find boundaries between phases for vertical lines
        phase_boundaries = []
        last_phase = None
        for i, phase in enumerate(phases):
            if phase != last_phase:
                if i > 0:
                    phase_boundaries.append((iterations[i], last_phase, phase))
                last_phase = phase
        
        # Plot vertical lines at phase transitions
        for boundary, from_phase, to_phase in phase_boundaries:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            plt.text(boundary + 0.5, max(fitness_values) * 0.95, 
                    f"{from_phase} → {to_phase}", 
                    rotation=90, verticalalignment='top')
        
        # Plot points by phase with larger markers at improvements
        for phase_name, color in phase_colors.items():
            # Get phase-specific data
            phase_indices = [i for i, p in enumerate(phases) if p == phase_name]
            if not phase_indices:
                continue
                
            phase_iterations = [iterations[i] for i in phase_indices]
            phase_fitness = [fitness_values[i] for i in phase_indices]
            
            # Find improvements within this phase
            improvements = []
            improvement_iterations = []
            
            if phase_iterations:
                best_so_far = float('inf')
                for i, fitness in zip(phase_iterations, phase_fitness):
                    if fitness < best_so_far:
                        best_so_far = fitness
                        improvements.append(fitness)
                        improvement_iterations.append(i)
            
            # Plot all phase points
            plt.plot(phase_iterations, phase_fitness, 'o', color=color, markersize=5, 
                    alpha=0.5, label=f"{phase_name} Progress")
            
            # Highlight improvements with larger markers
            if improvement_iterations:
                plt.plot(improvement_iterations, improvements, 'o', color=color, markersize=8,
                       label=f"{phase_name} Improvements")
        
        # Calculate improvement percentages for each phase
        phase_improvements = {}
        for phase_name in phase_colors:
            phase_entries = [entry for entry in self.history if entry['phase'] == phase_name]
            if phase_entries:
                initial = phase_entries[0]['fitness']
                final = phase_entries[-1]['fitness']
                improvement = (initial - final) / initial * 100 if initial > 0 else 0
                phase_improvements[phase_name] = improvement
        
        # Add improvement percentages to the title
        improvement_text = " | ".join([f"{phase}: {imp:.1f}%" for phase, imp in phase_improvements.items()])
        
        # Add overall stats
        if self.history:
            initial_fitness = self.history[0]['fitness']
            final_fitness = self.history[-1]['fitness']
            overall_improvement = (initial_fitness - final_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0
            
            plt.title(f'Phased Optimization Progress\nOverall improvement: {overall_improvement:.1f}% | {improvement_text}')
        else:
            plt.title('Phased Optimization Progress')
            
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (Distance)')
        plt.grid(True, alpha=0.3)
        
        # Add a detailed legend
        plt.legend(loc='upper right')
        
        # Improve layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to '{save_path}'")
            
        # Add annotations for significant improvements
        if self.history:
            # Find the top 3 largest improvements
            improvements = []
            for i in range(1, len(self.history)):
                prev_fitness = self.history[i-1]['fitness']
                curr_fitness = self.history[i]['fitness']
                if curr_fitness < prev_fitness:
                    improvement = (prev_fitness - curr_fitness) / prev_fitness * 100
                    improvements.append((i, improvement, self.history[i]['phase']))
            
            # Sort by improvement size
            improvements.sort(key=lambda x: x[1], reverse=True)
            
            # Annotate top improvements
            for idx, (i, imp_pct, phase) in enumerate(improvements[:3]):
                iter_num = self.history[i]['iteration']
                fitness = self.history[i]['fitness']
                plt.annotate(f"{imp_pct:.1f}% ({phase})", 
                           xy=(iter_num, fitness),
                           xytext=(10, 10 + idx * 20),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                           
        # Stretch y-axis slightly to make room for annotations
        y_min, y_max = plt.ylim()
        plt.ylim(y_min, y_max * 1.05)

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
    use_adaptive_params=True,
    use_diversity_management=True,
    use_solution_caching=True,
    use_advanced_local_search=True,
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
        use_adaptive_params: Enable adaptive parameter adjustments
        use_diversity_management: Enable diversity monitoring and intervention
        use_solution_caching: Enable solution caching for efficiency
        use_advanced_local_search: Enable 3-opt and other advanced local search
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
        use_adaptive_params=use_adaptive_params,
        use_diversity_management=use_diversity_management,
        use_solution_caching=use_solution_caching,
        use_advanced_local_search=use_advanced_local_search,
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
    print(f"Advanced features:")
    print(f"  Adaptive parameters: {'Enabled' if use_adaptive_params else 'Disabled'}")
    print(f"  Diversity management: {'Enabled' if use_diversity_management else 'Disabled'}")
    print(f"  Solution caching: {'Enabled' if use_solution_caching else 'Disabled'}")
    print(f"  Advanced local search: {'Enabled' if use_advanced_local_search else 'Disabled'}")
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
    
    # Print feature performance stats if used
    if use_adaptive_params and hasattr(solver, 'adaptive_params'):
        print("\nAdaptive Parameter Statistics:")
        for param, values in solver.adaptive_params.get_parameter_history().items():
            if len(values) > 1:  # Only show parameters that changed
                print(f"  {param}: initial={values[0]:.3f}, final={values[-1]:.3f}, " +
                      f"change={((values[-1]-values[0])/values[0]*100):.1f}%")
    
    if use_diversity_management and hasattr(solver, 'diversity_manager'):
        diversity_history = solver.diversity_manager.get_diversity_history()
        if diversity_history:
            print("\nDiversity Statistics:")
            print(f"  Average diversity: {sum(diversity_history)/len(diversity_history):.3f}")
            print(f"  Minimum diversity: {min(diversity_history):.3f}")
            print(f"  Diversity interventions: {len(solver.diversity_manager.get_intervention_history())}")
    
    if use_solution_caching and hasattr(solver, 'solution_cache'):
        print("\nSolution Cache Statistics:")
        print(f"  Cache hit rate: {solver.solution_cache.get_hit_rate():.1f}%")
        print(f"  Cache size: {solver.solution_cache.get_cache_size()} solutions")
    
    # Analyze phases performance
    if solver.history:
        phase_improvements = {}
        for phase in ['IGWO', 'GWO', 'GA']:
            phase_entries = [entry for entry in solver.history if entry['phase'] == phase]
            if phase_entries:
                initial = phase_entries[0]['fitness']
                final = phase_entries[-1]['fitness']
                improvement = (initial - final) / initial * 100 if initial > 0 else 0
                phase_improvements[phase] = improvement
                
        print("\nPhase Performance:")
        for phase, improvement in phase_improvements.items():
            print(f"  {phase}: {improvement:.1f}% improvement")
    
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

    # Test the main function with random cities
    # run_hybrid_phased(num_cities=20, population_size=50, max_iterations=200, 
    #                  igwo_share=0.33, gwo_share=0.33)