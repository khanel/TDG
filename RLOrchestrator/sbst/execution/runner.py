from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .models import ExecutionResult


class ExecutionError(RuntimeError):
    pass


def _which(tool: str) -> Optional[str]:
    return shutil.which(tool)


def run_command(
    command: List[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> ExecutionResult:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
        return ExecutionResult(
            command=list(command),
            cwd=str(cwd),
            exit_code=int(proc.returncode),
            timed_out=False,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(
            command=list(command),
            cwd=str(cwd),
            exit_code=None,
            timed_out=True,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
        )


def build_maven_command(*, goals: List[str]) -> List[str]:
    mvn = _which("mvn")
    if not mvn:
        raise ExecutionError("Maven executable 'mvn' not found on PATH")
    # Keep it explicit and debuggable; avoid -q by default to preserve logs.
    return [mvn, *goals]


def build_gradle_command(project_root: Path, *, tasks: List[str], use_wrapper: bool) -> List[str]:
    if use_wrapper:
        wrapper = project_root / "gradlew"
        if wrapper.exists():
            return [str(wrapper), *tasks]

    gradle = _which("gradle")
    if not gradle:
        raise ExecutionError("Gradle executable 'gradle' not found on PATH and no ./gradlew wrapper present")
    return [gradle, *tasks]
