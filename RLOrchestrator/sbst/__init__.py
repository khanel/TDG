"""SBST Test Data Generation (TDG) module.

This package will host the Java SBST TDG implementation (JUnit generation, execution,
coverage parsing) and its integration points with the existing orchestrator.

Design reference: docs/tdg_sbst_inheritance_plan.md
"""

from .adapter import SBSTAdapter

__all__ = ["SBSTAdapter"]
