from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    final_model_path: Path
    config_path: Path


def prepare_run_artifacts(
    *,
    mode: str,
    problem: str,
    model_output: str,
    session_id: int,
    args: Mapping[str, Any],
) -> RunArtifacts:
    final_model_path = Path(model_output).expanduser()
    if final_model_path.suffix != ".zip":
        final_model_path = final_model_path.with_suffix(".zip")

    run_dir = final_model_path.parent / f"{final_model_path.stem}_run{int(session_id)}"
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    config_path = run_dir / "run_config.json"

    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "mode": str(mode),
        "problem": str(problem),
        "session_id": int(session_id),
        "model_output": str(model_output),
        "final_model_path": str(final_model_path),
        "run_dir": str(run_dir),
        "args": _json_sanitize(dict(args)),
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return RunArtifacts(
        run_dir=run_dir,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        final_model_path=final_model_path,
        config_path=config_path,
    )


def _json_sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    return str(value)
