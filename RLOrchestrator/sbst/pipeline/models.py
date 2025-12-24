from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

BuildTool = Literal["auto", "maven", "gradle"]
ObjectiveGranularity = Literal["class", "method"]


@dataclass(frozen=True)
class SBSTConfig:
    """Configuration surface for the SBST adapter.

    Configuration surface for the SBST Java pipeline (discovery + generation +
    Maven/Gradle execution + JaCoCo parsing).

    Notes:
    - `dimension` is kept because current solvers expect fixed-length integer genes.
    - `project_root`/`targets` will become required once Maven/Gradle execution lands.
    """

    # Candidate representation parameters (current scaffold)
    dimension: int = 24
    seed: Optional[int] = 42

    # SBST execution parameters
    project_root: Optional[str] = None
    build_tool: BuildTool = "auto"
    targets: List[str] = field(default_factory=list)

    # Build tool invocation overrides
    gradle_use_wrapper: bool = True
    maven_goals: List[str] = field(default_factory=lambda: ["test", "jacoco:report"])
    gradle_tasks: List[str] = field(default_factory=lambda: ["test", "jacocoTestReport"])
    jacoco_xml_path: Optional[str] = None

    # Test generation controls (Stage-6)
    max_tests_per_candidate: int = 1
    max_actions_per_test: int = 5
    package_strategy: str = "match_target"  # match_target | fixed
    fixed_test_package: str = "generated.sbst"

    # Fitness gating across inheritance (Stage-7)
    gating_enabled: bool = True
    gating_complete_threshold: float = 0.99
    plateau_window: int = 50

    # Objective granularity (method-level objectives are opt-in).
    # - "class": existing behavior (class-level BRANCH coverage)
    # - "method": target one method at a time; objective identity includes receiver+declaring+signature
    objective_granularity: ObjectiveGranularity = "class"

    # Artifact management
    work_dir: str = "runs/sbst"

    # Failure handling / execution guardrails
    timeout_seconds: int = 300

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolved_work_dir(self) -> Path:
        # Keep relative by default so repo-local runs are easy to inspect.
        return Path(self.work_dir)
