"""Utilities for working with Maximum Cut benchmark instances."""

from .maxcut import MaxCutProblem, MaxCutSpec, generate_random_maxcut
from .solvers import (
    MaxCutGWOSolver,
    MaxCutGWOSolverConfig,
    MaxCutCMAESSolver,
    MaxCutCMAESSolverConfig,
    MaxCutQDSolver,
    MaxCutQDSolverConfig,
)

__all__ = [
    "MaxCutProblem",
    "MaxCutSpec",
    "generate_random_maxcut",
    "MaxCutGWOSolver",
    "MaxCutGWOSolverConfig",
    "MaxCutCMAESSolver",
    "MaxCutCMAESSolverConfig",
    "MaxCutQDSolver",
    "MaxCutQDSolverConfig",
]
