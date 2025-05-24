"""
Population Diversity Manager for Phased Hybrid Solver

This module implements mechanisms to monitor and maintain population diversity
throughout the optimization process to prevent premature convergence.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from Core.problem import Solution

class DiversityManager:
    """
    A system for monitoring and maintaining population diversity throughout
    the optimization process.
    """
    
    def __init__(self, diversity_threshold: float = 0.3, intervention_rate: float = 0.1):
        """
        Initialize the diversity manager.
        
        Args:
            diversity_threshold: Threshold below which diversity is considered too low.
            intervention_rate: Rate at which interventions are applied.
        """
        self.diversity_threshold = diversity_threshold
        self.intervention_rate = intervention_rate
        self.diversity_history = []
        self.intervention_history = []
    
    def measure_diversity(self, population: List[Solution]) -> float:
        """
        Measure the diversity of a population using route similarity metrics.
        
        Args:
            population: List of solution objects.
            
        Returns:
            Diversity score between 0.0 (no diversity) and 1.0 (maximum diversity).
        """
        if not population or len(population) < 2:
            return 1.0  # By definition, a single solution has maximum diversity
            
        n_individuals = len(population)
        n_cities = len(population[0].representation)
        total_similarity = 0
        
        # Calculate pairwise similarities
        for i in range(n_individuals):
            for j in range(i+1, n_individuals):
                # For TSP, we'll use edge similarity as our diversity metric
                # Convert routes to edge sets
                edges_i = self._route_to_edge_set(population[i].representation)
                edges_j = self._route_to_edge_set(population[j].representation)
                
                # Calculate Jaccard similarity (intersection over union)
                intersection = len(edges_i.intersection(edges_j))
                union = len(edges_i.union(edges_j))
                similarity = intersection / union if union > 0 else 0
                
                total_similarity += similarity
        
        # Calculate average similarity
        pairs = (n_individuals * (n_individuals - 1)) // 2
        avg_similarity = total_similarity / pairs if pairs > 0 else 0
        
        # Convert to diversity (1 - similarity)
        diversity = 1 - avg_similarity
        
        # Store in history
        self.diversity_history.append(diversity)
        
        return diversity
    
    def _route_to_edge_set(self, route: List[int]) -> set:
        """
        Convert a TSP route to a set of edges.
        
        Args:
            route: List of cities in visit order.
            
        Returns:
            Set of edges (as frozensets of city pairs).
        """
        edges = set()
        n = len(route)
        
        for i in range(n):
            # Add edge (current city, next city)
            # Using frozenset to make edges undirected
            edge = frozenset([route[i], route[(i+1) % n]])
            edges.add(edge)
            
        return edges
    
    def should_intervene(self, population: List[Solution]) -> bool:
        """
        Determine whether diversity intervention is needed.
        
        Args:
            population: List of solution objects.
            
        Returns:
            True if intervention is needed, False otherwise.
        """
        diversity = self.measure_diversity(population)
        return diversity < self.diversity_threshold
    
    def apply_diversity_intervention(
        self, 
        population: List[Solution],
        problem,
        best_solution: Optional[Solution] = None
    ) -> List[Solution]:
        """
        Apply diversity intervention to the population.
        
        Args:
            population: List of solution objects.
            problem: The problem instance.
            best_solution: The best solution found so far (optional).
            
        Returns:
            Modified population with increased diversity.
        """
        if not self.should_intervene(population):
            return population
            
        # Record intervention
        self.intervention_history.append(len(self.diversity_history) - 1)
        
        # Determine number of individuals to modify
        n_individuals = len(population)
        n_modify = max(2, int(n_individuals * self.intervention_rate))
        
        # Preserve best solutions
        if best_solution:
            # Ensure the best solution is in the population
            best_in_pop = False
            for sol in population:
                if np.array_equal(sol.representation, best_solution.representation):
                    best_in_pop = True
                    break
                    
            if not best_in_pop:
                # Replace a random solution with the best solution
                rand_idx = np.random.randint(0, n_individuals)
                population[rand_idx] = best_solution.copy()
        
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness)
        
        # Keep the best solutions unchanged
        n_elite = max(1, n_individuals // 10)  # Keep top 10% as elite
        
        # Apply different diversity interventions to non-elite solutions
        for i in range(n_elite, min(n_elite + n_modify, n_individuals)):
            # Choose a random diversity intervention strategy
            strategy = np.random.choice(['inversion', 'scramble', 'insertion'])
            
            # Apply the chosen strategy
            if strategy == 'inversion':
                # Inversion mutation: reverse a segment of the route
                route = population[i].representation.copy()
                n_cities = len(route)
                
                # Select random segment (preserving city 1 at position 0)
                start = 1 + np.random.randint(0, n_cities - 3)
                end = 1 + np.random.randint(start, n_cities - 1)
                
                # Reverse the segment
                route[start:end+1] = route[start:end+1][::-1]
                
                # Create and evaluate new solution
                new_sol = Solution(route, problem)
                new_sol.evaluate()
                population[i] = new_sol
                
            elif strategy == 'scramble':
                # Scramble mutation: randomly reorder a segment of the route
                route = population[i].representation.copy()
                n_cities = len(route)
                
                # Select random segment (preserving city 1 at position 0)
                start = 1 + np.random.randint(0, n_cities - 3)
                end = 1 + np.random.randint(start, n_cities - 1)
                
                # Scramble the segment
                segment = route[start:end+1].copy()
                np.random.shuffle(segment)
                route[start:end+1] = segment
                
                # Create and evaluate new solution
                new_sol = Solution(route, problem)
                new_sol.evaluate()
                population[i] = new_sol
                
            elif strategy == 'insertion':
                # Insertion mutation: remove a city and insert it elsewhere
                route = population[i].representation.copy()
                n_cities = len(route)
                
                # Select a random city (not the first one)
                city_idx = 1 + np.random.randint(0, n_cities - 1)
                city = route[city_idx]
                
                # Remove the city
                new_route = route[:city_idx] + route[city_idx+1:]
                
                # Choose a random insertion point (after city 1)
                insert_pos = 1 + np.random.randint(0, n_cities - 2)
                
                # Insert the city
                new_route = new_route[:insert_pos] + [city] + new_route[insert_pos:]
                
                # Create and evaluate new solution
                new_sol = Solution(new_route, problem)
                new_sol.evaluate()
                population[i] = new_sol
        
        return population
    
    def get_diversity_history(self) -> List[float]:
        """
        Get the history of diversity measurements.
        
        Returns:
            List of diversity values over time.
        """
        return self.diversity_history
    
    def get_intervention_history(self) -> List[int]:
        """
        Get the history of diversity interventions.
        
        Returns:
            List of indices in the diversity history where interventions occurred.
        """
        return self.intervention_history
