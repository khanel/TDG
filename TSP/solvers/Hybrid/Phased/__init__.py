"""
Phased Hybrid Optimization for TSP

This module implements a phased hybrid approach that applies three optimization
algorithms sequentially to solve the Traveling Salesperson Problem (TSP).

The three phases are:
1. IGWO (Improved Grey Wolf Optimization) - Initial exploration
2. GWO (Grey Wolf Optimization) - Continued exploration
3. GA (Genetic Algorithm) - Final exploitation

The entire population of solutions from each phase is passed to the next phase,
preserving diversity while enabling a comprehensive exploration-exploitation strategy.
"""

from .phased_solver import PhasedHybridSolver, run_hybrid_phased, run_phased_solver

__all__ = ["PhasedHybridSolver", "run_hybrid_phased", "run_phased_solver"]