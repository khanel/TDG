"""
Adaptive Parameter System for Phased Hybrid Solver

This module implements adaptive parameter adjustment mechanisms that
dynamically tune algorithm parameters based on search progress.
"""

import numpy as np
from typing import Dict, List, Any, Optional

class AdaptiveParameterSystem:
    """
    A system for dynamically adjusting algorithm parameters based on
    search progress and convergence behavior.
    """
    
    def __init__(
        self,
        initial_parameters: Dict[str, float],
        adaptation_rate: float = 0.1,
        min_values: Dict[str, float] = None,
        max_values: Dict[str, float] = None
    ):
        """
        Initialize the adaptive parameter system.
        
        Args:
            initial_parameters: Dictionary of parameter names and initial values.
            adaptation_rate: Rate at which parameters are adjusted (0.0 to 1.0).
            min_values: Dictionary of minimum allowed values for each parameter.
            max_values: Dictionary of maximum allowed values for each parameter.
        """
        self.parameters = initial_parameters.copy()
        self.adaptation_rate = adaptation_rate
        self.min_values = min_values or {}
        self.max_values = max_values or {}
        
        # Initialize default bounds for any missing parameters
        for param in self.parameters:
            if param not in self.min_values:
                self.min_values[param] = 0.0
            if param not in self.max_values:
                self.max_values[param] = 1.0
                
        # History of parameter values
        self.parameter_history = {param: [value] for param, value in self.parameters.items()}
        
        # Performance tracking
        self.last_improvement = 0
        self.stagnation_counter = 0
        self.improvement_history = []
        
    def get_parameter(self, name: str) -> float:
        """
        Get the current value of a parameter.
        
        Args:
            name: Name of the parameter.
            
        Returns:
            Current value of the parameter.
        """
        return self.parameters.get(name)
    
    def update_parameters(
        self,
        fitness_history: List[float],
        current_iteration: int,
        phase: str
    ):
        """
        Update parameters based on search progress.
        
        Args:
            fitness_history: List of best fitness values over iterations.
            current_iteration: The current iteration number.
            phase: Current algorithm phase ('IGWO', 'GWO', or 'GA').
        """
        if len(fitness_history) < 2:
            return
            
        # Calculate improvement rate over last 10 iterations or all if less
        window_size = min(10, len(fitness_history))
        recent_fitness = fitness_history[-window_size:]
        
        # Check if improvement has occurred
        if recent_fitness[-1] < recent_fitness[0]:
            # Improvement occurred
            improvement_rate = (recent_fitness[0] - recent_fitness[-1]) / recent_fitness[0]
            self.improvement_history.append(improvement_rate)
            self.last_improvement = current_iteration
            self.stagnation_counter = 0
            
            # Adjust parameters based on the phase and improvement
            if phase == 'IGWO':
                # If IGWO is improving rapidly, increase exploration parameters
                self._adjust_parameter('a_initial', improvement_rate * 0.5)
                self._adjust_parameter('k', improvement_rate * 0.5)
            elif phase == 'GWO':
                # If GWO is improving, balance between exploration and exploitation
                if improvement_rate > 0.05:  # Significant improvement
                    self._adjust_parameter('exploration_rate', 0.1)  # Slight increase in exploration
                else:
                    self._adjust_parameter('exploration_rate', -0.05)  # Decrease exploration slightly
            elif phase == 'GA':
                # If GA is improving, adjust mutation and crossover rates
                if improvement_rate > 0.05:  # Significant improvement
                    # Successful exploitation, maintain current parameters
                    pass
                else:
                    # Small improvements, increase mutation to escape potential local optima
                    self._adjust_parameter('mutation_rate', 0.1)
                    self._adjust_parameter('crossover_rate', -0.05)
        else:
            # No improvement, increment stagnation counter
            self.stagnation_counter += 1
            
            # If stagnation persists, make more aggressive parameter adjustments
            if self.stagnation_counter >= 20:
                if phase == 'IGWO':
                    self._adjust_parameter('a_initial', 0.3)  # Significant increase in exploration
                    self._adjust_parameter('k', 0.2)
                elif phase == 'GWO':
                    self._adjust_parameter('exploration_rate', 0.2)  # Increase exploration significantly
                elif phase == 'GA':
                    self._adjust_parameter('mutation_rate', 0.3)  # Increase mutation significantly
                    self._adjust_parameter('elitism_rate', -0.1)  # Decrease elitism to allow more exploration
                
                # Reset stagnation counter
                self.stagnation_counter = 0
    
    def _adjust_parameter(self, param_name: str, adjustment: float):
        """
        Adjust a parameter by the specified amount within bounds.
        
        Args:
            param_name: Name of the parameter to adjust.
            adjustment: Amount to adjust the parameter by (positive or negative).
        """
        if param_name not in self.parameters:
            return
            
        # Calculate new value with adaptive rate
        current_value = self.parameters[param_name]
        new_value = current_value + (adjustment * self.adaptation_rate)
        
        # Enforce bounds
        new_value = max(self.min_values.get(param_name, 0.0), new_value)
        new_value = min(self.max_values.get(param_name, 1.0), new_value)
        
        # Update parameter
        self.parameters[param_name] = new_value
        self.parameter_history[param_name].append(new_value)
    
    def get_parameter_history(self) -> Dict[str, List[float]]:
        """
        Get the history of parameter values.
        
        Returns:
            Dictionary of parameter names and their value histories.
        """
        return self.parameter_history
    
    def reset_stagnation(self):
        """Reset the stagnation counter, typically called when changing phases."""
        self.stagnation_counter = 0
