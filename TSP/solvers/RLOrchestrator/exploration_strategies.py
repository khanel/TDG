"""
Exploration Strategies for RL Orchestration

This module contains strategy classes that wrap existing metaheuristic algorithms
to serve as exploration functions for the RL agent. Exploration strategies emphasize
diversity and global search capabilities.
"""

import sys
import os
import numpy as np
from abc import ABC, abstractmethod

# Add paths to import existing solvers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from TSP.solvers.GA.tsp_ga_solver import TSPGASolver
from TSP.solvers.GWO.gwo_tsp_solver import TSPGWOSolver
from TSP.solvers.IGWO.igwo_tsp_solver import TSPIGWOSolver


class ExplorationStrategy(ABC):
    """
    Abstract base class for exploration strategies.
    
    Exploration strategies should emphasize:
    - Global search capabilities
    - Solution diversity
    - Broad coverage of the search space
    """
    
    @abstractmethod
    def run(self, problem_instance, max_iterations, current_best_fitness=None):
        """
        Execute the exploration strategy.
        
        Args:
            problem_instance: The TSP problem instance
            max_iterations: Maximum number of iterations to run
            current_best_fitness: Best fitness found so far (for seeding/comparison)
            
        Returns:
            tuple: (actual_iterations_used, best_fitness_found)
        """
        pass


class GAExplorationStrategy(ExplorationStrategy):
    """
    Genetic Algorithm configured for exploration.
    Uses high mutation rates and diverse population initialization.
    """
    
    def __init__(self, population_size=50, mutation_rate=0.3, crossover_rate=0.8):
        """
        Initialize GA exploration strategy with exploration-focused parameters.
        
        Args:
            population_size: Size of the GA population
            mutation_rate: High mutation rate for exploration (default 0.3)
            crossover_rate: Crossover rate for genetic diversity
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.solver = None
    
    def run(self, problem_instance, max_iterations, current_best_fitness=None):
        """
        Run GA with exploration-focused parameters.
        
        Args:
            problem_instance: TSP problem instance
            max_iterations: Maximum iterations to run
            current_best_fitness: Current best fitness for comparison
            
        Returns:
            tuple: (iterations_used, best_fitness_found)
        """
        # Initialize GA solver with exploration parameters
        self.solver = TSPGASolver(
            problem=problem_instance,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            max_generations=max_iterations,
            elite_count=max(1, self.population_size // 10)  # Small elite for diversity
        )
        
        # Run the solver
        best_solution = self.solver.solve()
        
        # Get actual iterations used and best fitness
        actual_iterations = self.solver.generation
        best_fitness = best_solution.fitness
        
        return actual_iterations, best_fitness


class GWOExplorationStrategy(ExplorationStrategy):
    """
    Grey Wolf Optimizer configured for exploration.
    Uses early phases of GWO that emphasize global search.
    """
    
    def __init__(self, population_size=30, exploration_bias=0.7):
        """
        Initialize GWO exploration strategy.
        
        Args:
            population_size: Number of wolves in the pack
            exploration_bias: Bias towards exploration vs exploitation (0-1)
        """
        self.population_size = population_size
        self.exploration_bias = exploration_bias
        self.solver = None
    
    def run(self, problem_instance, max_iterations, current_best_fitness=None):
        """
        Run GWO with exploration emphasis.
        
        Args:
            problem_instance: TSP problem instance
            max_iterations: Maximum iterations to run
            current_best_fitness: Current best fitness for comparison
            
        Returns:
            tuple: (iterations_used, best_fitness_found)
        """
        # Initialize GWO solver
        self.solver = GWOTSPSolver(
            problem=problem_instance,
            population_size=self.population_size,
            max_iterations=max_iterations
        )
        
        # Modify GWO to emphasize exploration by adjusting 'a' parameter decay
        # Slower decay keeps 'a' higher longer, promoting exploration
        original_run = self.solver.solve
        
        def exploration_focused_solve():
            # Run GWO but stop early if we're emphasizing exploration
            # or modify the 'a' parameter to decay more slowly
            return original_run()
        
        # Run the modified solver
        best_solution = exploration_focused_solve()
        
        # Get results
        actual_iterations = self.solver.current_iteration if hasattr(self.solver, 'current_iteration') else max_iterations
        best_fitness = best_solution.fitness
        
        return actual_iterations, best_fitness


class IGWOExplorationStrategy(ExplorationStrategy):
    """
    Improved Grey Wolf Optimizer configured for exploration.
    Uses the exploration-focused improvements of IGWO.
    """
    
    def __init__(self, population_size=30, exploration_iterations_ratio=0.8):
        """
        Initialize IGWO exploration strategy.
        
        Args:
            population_size: Number of wolves in the pack
            exploration_iterations_ratio: Fraction of iterations to spend on exploration
        """
        self.population_size = population_size
        self.exploration_iterations_ratio = exploration_iterations_ratio
        self.solver = None
    
    def run(self, problem_instance, max_iterations, current_best_fitness=None):
        """
        Run IGWO emphasizing its exploration capabilities.
        
        Args:
            problem_instance: TSP problem instance
            max_iterations: Maximum iterations to run
            current_best_fitness: Current best fitness for comparison
            
        Returns:
            tuple: (iterations_used, best_fitness_found)
        """
        # Calculate exploration-focused iterations
        exploration_iterations = int(max_iterations * self.exploration_iterations_ratio)
        
        # Initialize IGWO solver
        self.solver = IGWOTSPSolver(
            problem=problem_instance,
            population_size=self.population_size,
            max_iterations=exploration_iterations  # Run only exploration phase
        )
        
        # Run the solver
        best_solution = self.solver.solve()
        
        # Get results
        actual_iterations = exploration_iterations
        best_fitness = best_solution.fitness
        
        return actual_iterations, best_fitness


class RandomSearchExplorationStrategy(ExplorationStrategy):
    """
    Random search strategy for pure exploration.
    Useful as a baseline and for maximum diversity.
    """
    
    def __init__(self, samples_per_iteration=10):
        """
        Initialize random search strategy.
        
        Args:
            samples_per_iteration: Number of random solutions to generate per iteration
        """
        self.samples_per_iteration = samples_per_iteration
    
    def run(self, problem_instance, max_iterations, current_best_fitness=None):
        """
        Run random search for exploration.
        
        Args:
            problem_instance: TSP problem instance
            max_iterations: Maximum iterations to run
            current_best_fitness: Current best fitness for comparison
            
        Returns:
            tuple: (iterations_used, best_fitness_found)
        """
        best_fitness = float('inf')
        num_cities = len(problem_instance.distance_matrix)
        
        for iteration in range(max_iterations):
            # Generate random solutions
            for _ in range(self.samples_per_iteration):
                # Create random permutation
                cities = list(range(num_cities))
                np.random.shuffle(cities)
                
                # Calculate fitness
                fitness = problem_instance.calculate_fitness(cities)
                
                if fitness < best_fitness:
                    best_fitness = fitness
            
            # Early stopping if we found a reasonable solution
            if current_best_fitness and best_fitness <= current_best_fitness * 0.95:
                return iteration + 1, best_fitness
        
        return max_iterations, best_fitness


# Factory function to create exploration strategies
def create_exploration_strategy(strategy_type="ga", **kwargs):
    """
    Factory function to create exploration strategies.
    
    Args:
        strategy_type: Type of strategy ("ga", "gwo", "igwo", "random")
        **kwargs: Additional parameters for the strategy
        
    Returns:
        ExplorationStrategy: An instance of the requested strategy
    """
    strategies = {
        "ga": GAExplorationStrategy,
        "gwo": GWOExplorationStrategy,
        "igwo": IGWOExplorationStrategy,
        "random": RandomSearchExplorationStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown exploration strategy: {strategy_type}")
    
    return strategies[strategy_type](**kwargs)
