import numpy as np
import math
from typing import Callable, Tuple, List, Optional
import matplotlib.pyplot as plt

class IGWO:
    """
    Improved Gray Wolf Optimizer (IGWO) implementation based on the paper:
    "An improved gray wolf optimization algorithm solving to functional optimization and engineering design problems"
    
    Features:
    - Lens imaging reverse learning for initial population
    - Nonlinear control parameter convergence strategy
    - Modified search mechanism inspired by TSA and PSO
    """
    
    def __init__(self, 
                 objective_func: Callable[[np.ndarray], float],
                 dim: int,
                 lb: float,
                 ub: float,
                 population_size: int = 30,
                 max_iter: int = 1000,
                 k: float = 0.3,
                 a_initial: float = 2.0,
                 a_end: float = 0.0,
                 b1: float = 0.5,
                 b2: float = 0.5):
        """
        Initialize the IGWO optimizer.
        
        Parameters:
        - objective_func: The objective function to minimize
        - dim: Dimension of the problem
        - lb: Lower bound for each dimension
        - ub: Upper bound for each dimension
        - population_size: Number of wolves in the population
        - max_iter: Maximum number of iterations
        - k: Nonlinear modulation index for control parameter
        - a_initial: Initial value of parameter a
        - a_end: Final value of parameter a
        - b1: Individual memory coefficient
        - b2: Group communication coefficient
        """
        self.objective_func = objective_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.population_size = population_size
        self.max_iter = max_iter
        self.k = k
        self.a_initial = a_initial
        self.a_end = a_end
        self.b1 = b1
        self.b2 = b2
        
        # Initialize population and fitness
        self.population = None
        self.fitness = None
        self.alpha = None
        self.beta = None
        self.delta = None
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        
        # For modified search mechanism
        self.pbest_pos = None
        self.pbest_score = None
        
        # For tracking convergence
        self.convergence_curve = np.zeros(max_iter)
        
    def initialize_population(self):
        """Initialize the population with lens imaging reverse learning."""
        # Standard random initialization
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        
        # Apply lens imaging reverse learning
        k_init = (1 + (1 / self.max_iter)**0.5)**8  # Initial k value
        reverse_population = self.lens_imaging_reverse(self.population, k_init)
        
        # Combine and select best
        combined_pop = np.vstack((self.population, reverse_population))
        fitness = np.array([self.objective_func(ind) for ind in combined_pop])
        
        # Select top population_size individuals
        best_indices = np.argsort(fitness)[:self.population_size]
        self.population = combined_pop[best_indices]
        self.fitness = fitness[best_indices]
        
        # Initialize personal best
        self.pbest_pos = self.population.copy()
        self.pbest_score = self.fitness.copy()
        
    def lens_imaging_reverse(self, population: np.ndarray, k: float) -> np.ndarray:
        """
        Apply lens imaging reverse learning to generate reverse population.
        
        Parameters:
        - population: Current population
        - k: Scaling factor of the lens
        
        Returns:
        - Reverse population
        """
        min_vals = np.min(population, axis=0)
        max_vals = np.max(population, axis=0)
        
        # Calculate reverse positions
        reverse_pop = (min_vals + max_vals)/2 + (min_vals + max_vals)/(2*k) - population/k
        
        # Ensure bounds are respected
        reverse_pop = np.clip(reverse_pop, self.lb, self.ub)
        
        return reverse_pop
    
    def update_control_parameter(self, t: int) -> float:
        """
        Update the nonlinear control parameter a.
        
        Parameters:
        - t: Current iteration
        
        Returns:
        - Updated a value
        """
        return (self.a_initial - self.a_end) * np.exp(-(t**2)/(self.k * self.max_iter)**2) + self.a_end
    
    def update_position(self, t: int):
        """
        Update the position of wolves using the modified search mechanism.
        
        Parameters:
        - t: Current iteration
        """
        a = self.update_control_parameter(t)
        
        for i in range(self.population_size):
            # Update A and C parameters
            A1 = 2 * a * np.random.rand(self.dim) - a
            C1 = 2 * np.random.rand(self.dim)
            
            A2 = 2 * a * np.random.rand(self.dim) - a
            C2 = 2 * np.random.rand(self.dim)
            
            # Calculate new positions based on alpha and beta
            D_alpha = np.abs(C1 * self.alpha - self.population[i])
            X1 = self.alpha - A1 * D_alpha
            
            D_beta = np.abs(C2 * self.beta - self.population[i])
            X2 = self.beta - A2 * D_beta
            
            # Calculate weights based on fitness
            w1 = self.alpha_score / (self.alpha_score + self.beta_score)
            w2 = self.beta_score / (self.alpha_score + self.beta_score)
            
            # Modified position update equation
            r3 = np.random.rand(self.dim)
            r4 = np.random.rand(self.dim)
            perturbation = np.random.randn() + 2  # randn() + 2 to avoid division by zero
            
            self.population[i] = (w1 * X1 + w2 * X2 + 
                                (self.b1 * r3 * (self.pbest_pos[i] - self.population[i]) + 
                                 self.b2 * r4 * (X1 - self.population[i]))) / perturbation
            
            # Ensure bounds are respected
            self.population[i] = np.clip(self.population[i], self.lb, self.ub)
            
            # Update fitness
            self.fitness[i] = self.objective_func(self.population[i])
            
            # Update personal best
            if self.fitness[i] < self.pbest_score[i]:
                self.pbest_score[i] = self.fitness[i]
                self.pbest_pos[i] = self.population[i].copy()
    
    def update_leaders(self):
        """Update the alpha, beta, and delta wolves."""
        # Sort the population by fitness
        sorted_indices = np.argsort(self.fitness)
        
        # Update alpha, beta, delta
        self.alpha = self.population[sorted_indices[0]].copy()
        self.beta = self.population[sorted_indices[1]].copy()
        self.delta = self.population[sorted_indices[2]].copy()
        
        # Update their scores
        self.alpha_score = self.fitness[sorted_indices[0]]
        self.beta_score = self.fitness[sorted_indices[1]]
        self.delta_score = self.fitness[sorted_indices[2]]
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the IGWO optimization process.
        
        Returns:
        - best_solution: The best solution found
        - best_fitness: The fitness of the best solution
        """
        # Initialize population
        self.initialize_population()
        
        # Initial leader update
        self.update_leaders()
        
        # Optimization loop
        for t in range(self.max_iter):
            # Update positions
            self.update_position(t)
            
            # Update leaders
            self.update_leaders()
            
            # Store best fitness for convergence curve
            self.convergence_curve[t] = self.alpha_score
            
            # Print progress
            if t % 100 == 0:
                print(f"Iteration {t}: Best Fitness = {self.alpha_score}")
        
        return self.alpha, self.alpha_score
    
    def plot_convergence(self):
        """Plot the convergence curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'b', linewidth=2)
        plt.title('Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Define a test function (Sphere function)
    def sphere_function(x):
        return np.sum(x**2)
    
    # Set up the problem
    dim = 30
    lb = -100
    ub = 100
    
    # Create and run the optimizer
    igwo = IGWO(objective_func=sphere_function, 
                dim=dim, 
                lb=lb, 
                ub=ub,
                population_size=30,
                max_iter=500)
    
    best_solution, best_fitness = igwo.optimize()
    
    print("\nOptimization Results:")
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")
    
    # Plot convergence
    igwo.plot_convergence()