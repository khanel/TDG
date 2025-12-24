from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


def locate_jacoco_xml(
    project_root: Path,
    *,
    build_tool: str,
    override_path: Optional[str] = None,
) -> Optional[Path]:
    """Locate a JaCoCo XML report.

    Strategy:
    1) If override_path is provided and exists, use it.
    2) Try conventional Maven/Gradle locations.
    """

    if override_path:
        p = Path(override_path)
        if not p.is_absolute():
            p = project_root / p
        if p.exists() and p.is_file():
            return p

    candidates: List[Path] = []

    if build_tool == "maven":
        candidates.append(project_root / "target" / "site" / "jacoco" / "jacoco.xml")
    elif build_tool == "gradle":
        candidates.append(project_root / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml")

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None
