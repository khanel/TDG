"""SBST evaluation pipeline.

This package will grow into the full SBST TDG pipeline (discovery, generation,
execution, coverage parsing, caching).

Stage-1 provides a stable contract + artifact writing while the evaluation metric is
still a surrogate.
"""

from .models import SBSTConfig
from .pipeline import SBSTPipeline, SBSTEvaluationResult

__all__ = ["SBSTConfig", "SBSTPipeline", "SBSTEvaluationResult"]
