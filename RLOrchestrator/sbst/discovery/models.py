from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

BuildTool = Literal["maven", "gradle"]


@dataclass(frozen=True)
class TypeId:
    fqn: str


@dataclass
class HierarchyNode:
    type_id: TypeId
    parent_type_id: Optional[TypeId] = None
    is_external_parent: bool = False
    children: List[TypeId] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectUnderTest:
    root_path: str
    build_tool: BuildTool
    source_roots: List[str]
    test_roots: List[str]
    target_classes: List[str]

    # Lightly structured hierarchy representation: {child_fqn: node}
    hierarchy: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Optional method catalog. Each entry is a dict shaped like:
    # {
    #   "receiver_type_id": <fqn>,
    #   "declaring_type_id": <fqn>,
    #   "method_name": <name>,
    #   "jvm_descriptor": <descriptor>,
    #   "inherited": <bool>,
    #   "visibility": "public"|"protected"|"package"|"private"
    # }
    # This is additive metadata used for method-level objectives.
    callable_methods: List[Dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json_dict(payload: Dict[str, Any]) -> "ProjectUnderTest":
        return ProjectUnderTest(
            root_path=str(payload.get("root_path")),
            build_tool=str(payload.get("build_tool")),
            source_roots=list(payload.get("source_roots") or []),
            test_roots=list(payload.get("test_roots") or []),
            target_classes=list(payload.get("target_classes") or []),
            hierarchy=dict(payload.get("hierarchy") or {}),
            callable_methods=list(payload.get("callable_methods") or []),
        )

    @property
    def root(self) -> Path:
        return Path(self.root_path)
