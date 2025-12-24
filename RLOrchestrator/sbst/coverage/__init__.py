"""Coverage parsing and coverage-driven fitness utilities."""

from .jacoco_xml import CoverageParseError, method_key, parse_jacoco_xml
from .models import CoverageReport, CoverageSummary

__all__ = ["CoverageParseError", "CoverageReport", "CoverageSummary", "method_key", "parse_jacoco_xml"]
