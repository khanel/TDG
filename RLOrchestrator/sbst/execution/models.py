from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExecutionResult:
    command: List[str]
    cwd: str
    exit_code: Optional[int]
    timed_out: bool
    stdout: str
    stderr: str

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return (not self.timed_out) and (self.exit_code == 0)
