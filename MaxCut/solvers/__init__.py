"""Solvers tailored for Max-Cut instances."""

from .GWO.gwo_maxcut_solver import MaxCutGWOSolver, MaxCutGWOSolverConfig
from .CMAES.cmaes_maxcut_solver import MaxCutCMAESSolver, MaxCutCMAESSolverConfig
from .QD.maxcut_qd_solver import MaxCutQDSolver, MaxCutQDSolverConfig

__all__ = [
    "MaxCutGWOSolver",
    "MaxCutGWOSolverConfig",
    "MaxCutCMAESSolver",
    "MaxCutCMAESSolverConfig",
    "MaxCutQDSolver",
    "MaxCutQDSolverConfig",
]
