from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .models import GeneratedTest


def write_tests_to_directory(tests: Iterable[GeneratedTest], base_dir: Path) -> List[Path]:
    """Write GeneratedTest sources into a Java source tree.

    `base_dir` should be a test source root (e.g., <project>/src/test/java).

    Returns paths written.
    """

    written: List[Path] = []
    base_dir.mkdir(parents=True, exist_ok=True)

    for t in tests:
        pkg_path = base_dir / Path(t.package.replace(".", "/"))
        pkg_path.mkdir(parents=True, exist_ok=True)
        path = pkg_path / f"{t.class_name}.java"
        path.write_text(t.source, encoding="utf-8")
        written.append(path)

    return written


def delete_paths(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            continue
