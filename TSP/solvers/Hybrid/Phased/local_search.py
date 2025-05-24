"""
Advanced Local Search Techniques for TSP

This module implements advanced local search techniques for TSP solutions,
including 3-opt local search and other enhancement methods.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from Core.problem import Solution

class AdvancedLocalSearch:
    """
    Implements advanced local search techniques for TSP solutions.
    """
    
    def __init__(self, tsp_problem):
        """
        Initialize the advanced local search module.
        
        Args:
            tsp_problem: The TSP problem instance
        """
        self.tsp_problem = tsp_problem
        
    def apply_2opt_improvement(self, solution):
        """
        Apply a 2-opt local improvement to a TSP solution.
        This checks for crossing edges and replaces them with non-crossing edges.
        
        Args:
            solution (Solution): The TSP solution to improve
            
        Returns:
            Solution: Improved solution, or the original if no improvement was found
        """
        if not solution:
            return solution
            
        # Create a copy to avoid modifying the original
        best_sol = solution.copy()
        best_fitness = best_sol.fitness
        
        # Get the route
        route = best_sol.representation.copy()
        n = len(route)
        
        improved = False
        
        # Get city coordinates for faster distance calculations
        coords = self.tsp_problem.city_coords
        
        # Systematic search for crossing edges
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                # Calculate the change in tour length if we swap the edges
                # Current edges: (i-1,i) and (j,j+1)
                # New edges after swap: (i-1,j) and (i,j+1)
                
                # Using direct distance calculation if coordinates are available
                if coords:
                    # Current edges
                    dist_current = (np.sqrt((coords[route[i-1]][0] - coords[route[i]][0])**2 + 
                                          (coords[route[i-1]][1] - coords[route[i]][1])**2) + 
                                   np.sqrt((coords[route[j]][0] - coords[route[j+1]][0])**2 + 
                                          (coords[route[j]][1] - coords[route[j+1]][1])**2))
                    
                    # New edges
                    dist_new = (np.sqrt((coords[route[i-1]][0] - coords[route[j]][0])**2 + 
                                       (coords[route[i-1]][1] - coords[route[j]][1])**2) + 
                                np.sqrt((coords[route[i]][0] - coords[route[j+1]][0])**2 + 
                                       (coords[route[i]][1] - coords[route[j+1]][1])**2))
                else:
                    # Use distance matrix if coordinates aren't available
                    dist_matrix = self.tsp_problem.graph.dist_matrix
                    # Current edges
                    dist_current = (dist_matrix[route[i-1]][route[i]] + 
                                   dist_matrix[route[j]][route[j+1]])
                    
                    # New edges
                    dist_new = (dist_matrix[route[i-1]][route[j]] + 
                               dist_matrix[route[i]][route[j+1]])
                
                # If the new configuration is better, make the swap
                if dist_new < dist_current:
                    # Reverse the route between i and j
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
        
        # If we made an improvement, update the solution
        if improved:
            # Recalculate the fitness
            new_sol = Solution(route, solution.problem)
            new_sol.fitness = self.tsp_problem.evaluate(new_sol)
            
            if new_sol.fitness < best_fitness:
                return new_sol
                
        # Try random 2-opt if systematic approach didn't improve
        # This helps escape local optima
        attempts = min(20, n * 2)  # Limit attempts
        
        for _ in range(attempts):
            # Choose two random edges
            i = random.randint(1, n-3)
            j = random.randint(i+1, n-2)
            
            # Create a new route by reversing the segment between i and j
            new_route = route.copy()
            new_route[i:j+1] = reversed(new_route[i:j+1])
            
            # Create and evaluate the new solution
            new_sol = Solution(new_route, solution.problem)
            new_sol.fitness = self.tsp_problem.evaluate(new_sol)
            
            # Update if better
            if new_sol.fitness < best_fitness:
                return new_sol
        
        return best_sol  # Return the original if no improvement found
    
    def apply_3opt_improvement(self, solution):
        """
        Apply a 3-opt local improvement to a TSP solution.
        This advanced local search method tests all possible ways to reconnect a tour
        after removing 3 edges, potentially finding better solutions than 2-opt.
        
        Args:
            solution (Solution): The TSP solution to improve
            
        Returns:
            Solution: Improved solution, or the original if no improvement was found
        """
        if not solution:
            return solution
            
        # Create a copy to avoid modifying the original
        best_sol = solution.copy()
        best_fitness = best_sol.fitness
        
        # Get the route
        route = best_sol.representation.copy()
        n = len(route)
        
        # Get city coordinates for faster distance calculations
        coords = self.tsp_problem.city_coords
        
        # 3-opt is computationally expensive, so we'll use a limited approach
        # We'll test random triplets of edges
        max_attempts = min(20, n)  # Limit attempts due to computational complexity
        
        for _ in range(max_attempts):
            # Choose three random distinct positions in the route (avoiding the first city)
            indices = sorted(random.sample(range(1, n-1), 3))
            i, j, k = indices
            
            # There are several ways to reconnect a tour after removing 3 edges
            # We'll test each possibility and keep the best one
            
            # Original order: 0 -> i -> i+1 -> j -> j+1 -> k -> k+1 -> 0
            # Option 1: 0 -> i -> j+1 -> k -> j -> i+1 -> k+1 -> 0
            # Option 2: 0 -> i -> j+1 -> k -> j -> i+1 -> k+1 -> 0
            # Option 3: 0 -> i -> k+1 -> j+1 -> k -> j -> i+1 -> 0
            # Option 4: 0 -> i -> j -> k -> j+1 -> i+1 -> k+1 -> 0
            # There are 8 total ways to reconnect
            
            # We'll test a subset of the options for computational efficiency
            
            # Create alternative routes
            options = []
            
            # Original route
            options.append(route.copy())
            
            # Option 1: Reverse j+1 to k
            opt1 = route.copy()
            opt1[j+1:k+1] = list(reversed(opt1[j+1:k+1]))
            options.append(opt1)
            
            # Option 2: Reverse i+1 to j
            opt2 = route.copy()
            opt2[i+1:j+1] = list(reversed(opt2[i+1:j+1]))
            options.append(opt2)
            
            # Option 3: Reverse i+1 to j and j+1 to k
            opt3 = route.copy()
            opt3[i+1:j+1] = list(reversed(opt3[i+1:j+1]))
            opt3[j+1:k+1] = list(reversed(opt3[j+1:k+1]))
            options.append(opt3)
            
            # Option 4: Exchange segments
            opt4 = route.copy()
            segment1 = route[i+1:j+1].copy()
            segment2 = route[j+1:k+1].copy()
            opt4[i+1:i+1+len(segment2)] = segment2
            opt4[i+1+len(segment2):i+1+len(segment2)+len(segment1)] = segment1
            options.append(opt4)
            
            # Evaluate all options
            for opt_route in options:
                new_sol = Solution(opt_route, solution.problem)
                new_sol.fitness = self.tsp_problem.evaluate(new_sol)
                
                # If improved, keep it
                if new_sol.fitness < best_fitness:
                    best_sol = new_sol
                    best_fitness = new_sol.fitness
        
        return best_sol

# For backward compatibility, keep the original function
def _apply_3opt_improvement(self, solution):
    local_search = AdvancedLocalSearch(self.tsp_problem)
    return local_search.apply_3opt_improvement(solution)
