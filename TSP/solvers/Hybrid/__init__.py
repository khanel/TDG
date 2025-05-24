"""
Hybrid Metaheuristic Approaches for TSP

This package contains implementations of different hybrid metaheuristic approaches
for solving the Traveling Salesperson Problem (TSP).
"""

from .RoundRobin.round_robin import run_hybrid_round_robin
from .Parallel.parallel import run_hybrid_parallel
from .Phased.phased_solver import run_hybrid_phased, run_phased_solver

__all__ = ["run_hybrid_round_robin", "run_hybrid_parallel", "run_hybrid_phased", "run_phased_solver"]
