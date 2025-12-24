from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SBSTArtifactPaths:
    candidate_digest: str
    run_dir: Path
    config_json: Path
    discovery_json: Path
    tests_dir: Path
    build_logs_dir: Path
    coverage_dir: Path
    stdout_txt: Path
    stderr_txt: Path
    exit_code_json: Path
    command_txt: Path
    jacoco_xml: Path
    coverage_summary_json: Path
    candidate_json: Path


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def candidate_digest(candidate: Any) -> str:
    try:
        payload = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except TypeError:
        payload = repr(candidate).encode("utf-8")
    return sha256(payload).hexdigest()[:12]


def make_run_dir(base_dir: Path, *, solution_id: Optional[int], digest: str) -> Path:
    # Human-friendly + collision-resistant. Timestamp is for debuggability; digest keeps linkage.
    short_id = f"{int(solution_id):06d}" if solution_id is not None else "noid"
    run_name = f"{_utc_timestamp()}_{short_id}_{digest}"
    return base_dir / run_name


def prepare_artifacts(base_dir: Path, *, solution_id: Optional[int], candidate: Any) -> SBSTArtifactPaths:
    base_dir.mkdir(parents=True, exist_ok=True)
    digest = candidate_digest(candidate)
    run_dir = make_run_dir(base_dir, solution_id=solution_id, digest=digest)

    tests_dir = run_dir / "tests"
    build_logs_dir = run_dir / "build_logs"
    coverage_dir = run_dir / "coverage"

    for d in (run_dir, tests_dir, build_logs_dir, coverage_dir):
        d.mkdir(parents=True, exist_ok=True)

    return SBSTArtifactPaths(
        candidate_digest=digest,
        run_dir=run_dir,
        config_json=run_dir / "config.json",
        discovery_json=run_dir / "discovery.json",
        tests_dir=tests_dir,
        build_logs_dir=build_logs_dir,
        coverage_dir=coverage_dir,
        stdout_txt=build_logs_dir / "stdout.txt",
        stderr_txt=build_logs_dir / "stderr.txt",
        exit_code_json=build_logs_dir / "exit_code.json",
        command_txt=build_logs_dir / "command.txt",
        jacoco_xml=coverage_dir / "jacoco.xml",
        coverage_summary_json=coverage_dir / "coverage_summary.json",
        candidate_json=run_dir / "candidate.json",
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
