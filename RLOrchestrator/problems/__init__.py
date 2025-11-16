"""
Problem adapters and registry utilities.
"""

from .registry import (
    ProblemBundle,
    ProblemDefinition,
    SolverFactory,
    get_problem_adapter,
    get_problem_definition,
    get_problem_registry,
    instantiate_problem,
    list_problem_definitions,
    register_problem,
)

__all__ = [
    'ProblemBundle',
    'ProblemDefinition',
    'SolverFactory',
    'register_problem',
    'instantiate_problem',
    'list_problem_definitions',
    'get_problem_definition',
    'get_problem_registry',
    'get_problem_adapter',
]
