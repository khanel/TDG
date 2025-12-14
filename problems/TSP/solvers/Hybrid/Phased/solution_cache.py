"""
Solution Caching System for Phased Hybrid Solver

This module implements a caching system to avoid redundant solution evaluations
and to track the history of solutions explored during the optimization process.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set, FrozenSet
from Core.problem import Solution
import hashlib

class SolutionCache:
    """
    A caching system for TSP solutions to avoid redundant evaluations and
    track the history of solutions explored.
    """
    
    def __init__(self, max_cache_size: int = 10000):
        """
        Initialize the solution cache.
        
        Args:
            max_cache_size: Maximum number of solutions to cache.
        """
        self.max_cache_size = max_cache_size
        self.cache = {}  # Maps solution hash to (fitness, frequency)
        self.access_count = {}  # Maps solution hash to access count
        self.access_total = 0  # Total number of accesses
        self.hits = 0  # Number of cache hits
        self.misses = 0  # Number of cache misses
        
    def _hash_solution(self, solution: Solution) -> str:
        """
        Generate a hash for a TSP solution.
        
        Args:
            solution: The solution to hash.
            
        Returns:
            A hash string representing the solution.
        """
        # For TSP, we need to normalize the representation to handle rotational symmetry
        # A canonical form starts with city 1 and has the smallest city after 1 at the earliest position
        route = solution.representation
        if not route or route[0] != 1:
            return None  # Invalid solution
            
        # Convert route to string and hash
        route_str = ','.join(map(str, route))
        return hashlib.md5(route_str.encode()).hexdigest()
    
    def get(self, solution: Solution) -> Optional[float]:
        """
        Get a solution's fitness from the cache.
        
        Args:
            solution: The solution to look up.
            
        Returns:
            Fitness value if cached, None otherwise.
        """
        solution_hash = self._hash_solution(solution)
        if not solution_hash:
            self.misses += 1
            return None
            
        if solution_hash in self.cache:
            # Update access count
            self.access_count[solution_hash] = self.access_count.get(solution_hash, 0) + 1
            self.access_total += 1
            self.hits += 1
            
            # Update frequency
            fitness, frequency = self.cache[solution_hash]
            self.cache[solution_hash] = (fitness, frequency + 1)
            
            return fitness
        else:
            self.misses += 1
            return None
    
    def put(self, solution: Solution, fitness: float):
        """
        Add a solution to the cache.
        
        Args:
            solution: The solution to cache.
            fitness: The fitness value of the solution.
        """
        solution_hash = self._hash_solution(solution)
        if not solution_hash:
            return
            
        # If cache is full, remove least accessed items
        if len(self.cache) >= self.max_cache_size:
            self._prune_cache()
            
        # Add to cache or update if already exists
        if solution_hash in self.cache:
            _, frequency = self.cache[solution_hash]
            self.cache[solution_hash] = (fitness, frequency + 1)
        else:
            self.cache[solution_hash] = (fitness, 1)
            self.access_count[solution_hash] = 0
    
    def _prune_cache(self):
        """Remove least accessed items from the cache when it reaches max size."""
        # Sort by access count, then by frequency
        items_to_remove = len(self.cache) - int(self.max_cache_size * 0.8)  # Remove 20% of cache
        
        if items_to_remove <= 0:
            return
            
        # Sort items by access count and frequency
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: (self.access_count.get(x[0], 0), x[1][1])
        )
        
        # Remove least accessed/frequent items
        for i in range(items_to_remove):
            if i < len(sorted_items):
                solution_hash, _ = sorted_items[i]
                if solution_hash in self.cache:
                    del self.cache[solution_hash]
                if solution_hash in self.access_count:
                    del self.access_count[solution_hash]
    
    def get_hit_rate(self) -> float:
        """
        Get the cache hit rate.
        
        Returns:
            Cache hit rate as a percentage.
        """
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0
    
    def get_cache_size(self) -> int:
        """
        Get the current cache size.
        
        Returns:
            Number of solutions in the cache.
        """
        return len(self.cache)
    
    def get_frequent_solutions(self, limit: int = 10) -> List[Tuple[str, float, int]]:
        """
        Get the most frequently accessed solutions.
        
        Args:
            limit: Maximum number of solutions to return.
            
        Returns:
            List of tuples (solution hash, fitness, frequency).
        """
        # Sort by frequency (descending)
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1][1],
            reverse=True
        )
        
        # Return top solutions
        return [(h, f, freq) for h, (f, freq) in sorted_items[:limit]]
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self.access_count = {}
        self.access_total = 0
        self.hits = 0
        self.misses = 0
