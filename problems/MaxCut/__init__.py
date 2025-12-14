"""Utilities for working with Maximum Cut benchmark instances."""

from .maxcut import MaxCutProblem, MaxCutSpec, generate_random_maxcut

# Keep solver exports optional so package import doesn't fail when some solver
# implementations are not present.
from .solvers import MaxCutGWOSolver, MaxCutGWOSolverConfig

__all__ = [
    "MaxCutProblem",
    "MaxCutSpec",
    "generate_random_maxcut",
    "MaxCutGWOSolver",
    "MaxCutGWOSolverConfig",
]
