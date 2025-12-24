from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set


_EXCLUDED_DIR_NAMES: Set[str] = {"target", "build", "out", ".gradle", ".idea", ".git"}


def discovery_input_files(project_root: Path, source_roots: List[Path]) -> List[Path]:
    """Files that should invalidate discovery results (mtime-first then hash).

    MVP inputs per master plan:
    - Build files in root
    - Any .java under production source roots
    """
    build_files = [
        project_root / "pom.xml",
        project_root / "build.gradle",
        project_root / "build.gradle.kts",
        project_root / "settings.gradle",
        project_root / "settings.gradle.kts",
        project_root / "gradle.properties",
    ]

    files: List[Path] = [p for p in build_files if p.exists() and p.is_file()]
    for r in source_roots:
        files.extend(list(_iter_java_files(r)))
    # Keep stable order
    return sorted({p.resolve() for p in files}, key=lambda p: str(p))


def execution_input_files(project_root: Path, source_roots: List[Path]) -> List[Path]:
    """Files that should invalidate execution/coverage results.

    MVP inputs per master plan:
    - Everything from discovery inputs
    - (Future) generated test sources

    For now, we reuse discovery inputs because generation isn't wired yet.
    """
    return discovery_input_files(project_root, source_roots)


def _iter_java_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return

    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            for child in cur.iterdir():
                if child.is_dir():
                    if child.name in _EXCLUDED_DIR_NAMES:
                        continue
                    stack.append(child)
                elif child.is_file() and child.suffix == ".java":
                    yield child
        except OSError:
            continue
