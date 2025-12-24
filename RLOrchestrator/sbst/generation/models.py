from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class GeneratedTest:
    package: str
    class_name: str
    source: str

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)
