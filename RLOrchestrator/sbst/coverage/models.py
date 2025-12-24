from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CoverageSummary:
    branches_covered: int
    branches_missed: int

    @property
    def total_branches(self) -> int:
        return int(self.branches_covered) + int(self.branches_missed)

    @property
    def coverage_fraction(self) -> Optional[float]:
        total = self.total_branches
        if total <= 0:
            return None
        return float(self.branches_covered) / float(total)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self) | {"coverage_fraction": self.coverage_fraction}


@dataclass(frozen=True)
class CoverageReport:
    overall: CoverageSummary
    by_class: Dict[str, CoverageSummary] = field(default_factory=dict)
    by_method: Dict[str, CoverageSummary] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.to_json_dict(),
            "by_class": {k: v.to_json_dict() for k, v in self.by_class.items()},
            "by_method": {k: v.to_json_dict() for k, v in self.by_method.items()},
        }
