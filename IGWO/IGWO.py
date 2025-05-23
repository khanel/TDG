import numpy as np
import math
from typing import Callable, Tuple, List, Optional
import matplotlib.pyplot as plt

from Core.search_algorithm import SearchAlgorithm
from Core.problem import Solution

class IGWO(SearchAlgorithm):
    def __init__(self, problem, population_size, max_iterations=1000, k=0.3, a_initial=2.0, a_end=0.0, b1=0.5, b2=0.5, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.max_iterations = max_iterations
        self.k = k
        self.a_initial = a_initial
        self.a_end = a_end
        self.b1 = b1
        self.b2 = b2
        self.iteration = 0
        self.pbest: list[Solution] = []
        self.pbest_score: list[float] = []
        self.convergence_curve = []
        
    def initialize(self):
        # Use problem's initial population of Solution objects
        self.population = self.problem.get_initial_population(self.population_size)
        for sol in self.population:
            sol.evaluate()
        self.pbest = [Solution(sol.representation.copy(), sol.problem) for sol in self.population]
        self.pbest_score = [sol.fitness for sol in self.population]
        self._update_best_solution()
        
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
    
    def step(self):
        a = self.update_control_parameter(self.iteration)
        # Find alpha and beta wolves
        fitness = np.array([sol.fitness if sol.fitness is not None else sol.evaluate() for sol in self.population])
        idx = np.argsort(fitness)
        alpha = self.population[idx[0]]
        beta = self.population[idx[1]]
        # Update positions
        new_population = []
        for i, wolf in enumerate(self.population):
            A1 = 2 * a * np.random.rand(*np.shape(wolf.representation)) - a
            C1 = 2 * np.random.rand(*np.shape(wolf.representation))
            A2 = 2 * a * np.random.rand(*np.shape(wolf.representation)) - a
            C2 = 2 * np.random.rand(*np.shape(wolf.representation))
            D_alpha = np.abs(C1 * alpha.representation - wolf.representation)
            X1 = alpha.representation - A1 * D_alpha
            D_beta = np.abs(C2 * beta.representation - wolf.representation)
            X2 = beta.representation - A2 * D_beta
            w1 = fitness[idx[0]] / (fitness[idx[0]] + fitness[idx[1]])
            w2 = fitness[idx[1]] / (fitness[idx[0]] + fitness[idx[1]])
            r3 = np.random.rand(*np.shape(wolf.representation))
            r4 = np.random.rand(*np.shape(wolf.representation))
            perturbation = np.random.randn() + 2
            # Use pbest for memory
            pbest_vec = self.pbest[i].representation
            new_repr = (w1 * X1 + w2 * X2 + (self.b1 * r3 * (pbest_vec - wolf.representation) + self.b2 * r4 * (X1 - wolf.representation))) / perturbation
            # Boundary handling
            info = self.problem.get_problem_info()
            lb = np.array(info.get('lower_bounds', -np.inf))
            ub = np.array(info.get('upper_bounds', np.inf))
            new_repr = np.clip(new_repr, lb, ub)
            new_sol = Solution(new_repr, self.problem)
            new_sol.evaluate()
            # Update pbest
            if new_sol.fitness < self.pbest_score[i]:
                self.pbest[i] = Solution(new_sol.representation.copy(), new_sol.problem)
                self.pbest_score[i] = new_sol.fitness
            new_population.append(new_sol)
        self.population = new_population
        self._update_best_solution()
        self.iteration += 1
        self.convergence_curve.append(self.best_solution.fitness)
    
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