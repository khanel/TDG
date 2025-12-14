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
        self.pbest = [sol.copy() for sol in self.population]
        self.pbest_score = [sol.fitness for sol in self.population]
        self._update_best_solution()
        
    def update_control_parameter(self, t: int) -> float:
        """
        Update the nonlinear control parameter a.
        
        Parameters:
        - t: Current iteration
        
        Returns:
        - Updated a value
        """
        return (self.a_initial - self.a_end) * np.exp(-(t**2)/(self.k * self.max_iterations)**2) + self.a_end
    
    def step(self):
        # Check problem type
        problem_info = self.problem.get_problem_info()
        problem_type = problem_info.get('problem_type', 'continuous')
        
        if problem_type == 'discrete':
            # For discrete problems like TSP, use a discrete adaptation of IGWO
            self._discrete_step()
        else:
            # For continuous problems, use the standard IGWO
            self._continuous_step()
            
        self._update_best_solution()
        self.iteration += 1
        
    def _discrete_step(self):
        """
        Adaptation of IGWO for discrete problems like TSP.
        """
        # Make sure pbest has the same length as population
        if len(self.pbest) != len(self.population):
            print(f"Resetting pbest: {len(self.pbest)} != {len(self.population)}")
            self.pbest = [sol.copy() for sol in self.population]
            self.pbest_score = [sol.fitness for sol in self.population]
            
        # Sort wolves by fitness
        self.population.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        
        # Alpha and beta are the two best solutions
        alpha, beta = self.population[0], self.population[1]
        
        # Update control parameter
        a = self.update_control_parameter(self.iteration)
        
        new_population = []
        for i, wolf in enumerate(self.population):
            # Create a new solution based on alpha, beta, and pbest
            if np.random.rand() < a:  # Higher probability in early iterations
                # Apply crossover with leaders
                if np.random.rand() < 0.5:
                    leader = alpha
                else:
                    leader = beta
                    
                # Create a new solution through crossover
                new_repr = self._discrete_crossover(wolf.representation, leader.representation)
                
                # Apply a small perturbation (mutation)
                if np.random.rand() < self.b1:
                    new_repr = self._discrete_mutate(new_repr)
                    
                # Apply crossover with personal best with probability b2
                if np.random.rand() < self.b2:
                    new_repr = self._discrete_crossover(new_repr, self.pbest[i].representation)
            else:
                # More exploration in later iterations
                new_repr = self._discrete_mutate(wolf.representation.copy())
            
            # Create and evaluate new solution
            new_sol = Solution(new_repr, self.problem)
            new_sol.evaluate()
            
            # Update pbest
            if new_sol.fitness < self.pbest_score[i]:
                self.pbest[i] = Solution(new_sol.representation.copy(), new_sol.problem)
                self.pbest_score[i] = new_sol.fitness
                
            new_population.append(new_sol)
            
        self.population = new_population
        
    def _discrete_crossover(self, wolf_repr, leader_repr):
        """Crossover operation for discrete problems like TSP."""
        # Create a new solution that follows the leader partially
        n = len(wolf_repr)
        # Start with city 1 (fixed)
        new_repr = [1]
        
        # Copy a random segment from the leader (maintaining relative order)
        segment_length = np.random.randint(1, n // 2)
        start_pos = np.random.randint(1, n - segment_length)
        segment = leader_repr[start_pos:start_pos + segment_length]
        
        # Add cities from the segment that aren't already in new_repr
        for city in segment:
            if city not in new_repr:
                new_repr.append(city)
        
        # Add remaining cities from wolf in their original order
        for city in wolf_repr:
            if city not in new_repr:
                new_repr.append(city)
                
        return new_repr
        
    def _discrete_mutate(self, representation):
        """Mutation operation for discrete problems like TSP."""
        # Apply a random swap mutation
        new_repr = representation.copy()
        n = len(new_repr)
        
        if n > 3:  # Need at least 3 cities to swap (since city 1 is fixed)
            # Choose how many swaps to perform (1-3)
            num_swaps = np.random.randint(1, min(4, n//2))
            
            for _ in range(num_swaps):
                # Swap two random cities (excluding city 1)
                i, j = np.random.choice(range(1, n), size=2, replace=False)
                new_repr[i], new_repr[j] = new_repr[j], new_repr[i]
                
        return new_repr
        
    def _continuous_step(self):
        """Original IGWO algorithm for continuous problems."""
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