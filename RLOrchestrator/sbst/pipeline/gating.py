from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..coverage.models import CoverageSummary
from ..discovery.models import ProjectUnderTest


@dataclass
class GatingStatus:
    objective_target: Optional[str]
    objective_coverage_fraction: float
    gated_sequence: List[str]
    index: int
    completed: Set[str]
    plateaued: Set[str]


class ParentFirstGating:
    """Parent-first gating with a plateau heuristic.

    This scheduler chooses a *single* class objective at a time.
    It advances to the next class when either:
    - coverage >= completion_threshold, or
    - no improvement for `plateau_window` observations.

    Classes with no branches (coverage_fraction is None) are auto-completed.
    """

    def __init__(self, *, completion_threshold: float = 0.99, plateau_window: int = 50):
        self._completion_threshold = float(completion_threshold)
        self._plateau_window = max(1, int(plateau_window))

        self._sequence: List[str] = []
        self._index: int = 0

        self._best: float = 0.0
        self._no_improve: int = 0

        self._completed: Set[str] = set()
        self._plateaued: Set[str] = set()

        self._project_key: Optional[Tuple[str, Tuple[str, ...]]] = None

    def reset(self) -> None:
        self._sequence = []
        self._index = 0
        self._best = 0.0
        self._no_improve = 0
        self._completed = set()
        self._plateaued = set()
        self._project_key = None

    def ensure_initialized(self, *, project: ProjectUnderTest, targets: List[str]) -> None:
        key = (str(project.root_path), tuple(sorted(targets)))
        if self._project_key == key and self._sequence:
            return

        # Reset state when project/targets change.
        self.reset()
        self._project_key = key
        self._sequence = compute_gating_sequence(project.hierarchy, targets)
        self._index = 0

    def current_target(self) -> Optional[str]:
        while 0 <= self._index < len(self._sequence) and self._sequence[self._index] in self._completed:
            self._index += 1
            self._best = 0.0
            self._no_improve = 0

        if 0 <= self._index < len(self._sequence):
            return self._sequence[self._index]
        return None

    def observe(self, *, target: Optional[str], coverage: Optional[CoverageSummary]) -> GatingStatus:
        """Update state based on observed coverage for the current objective target.

        `target` should match `current_target()` (or be None when ungated).
        """

        objective_cov = _coverage_fraction_or_none(coverage)

        # If ungated, we don't advance anything.
        if target is None:
            return self.status(objective_target=None, objective_cov=1.0)

        # Auto-complete no-branch targets.
        if objective_cov is None:
            self._completed.add(target)
            self._advance_after_completion_or_plateau()
            return self.status(objective_target=self.current_target(), objective_cov=1.0)

        # Completion check.
        if objective_cov >= self._completion_threshold:
            self._completed.add(target)
            self._advance_after_completion_or_plateau()
            return self.status(objective_target=self.current_target(), objective_cov=float(objective_cov))

        # Plateau logic.
        eps = 1e-12
        if objective_cov > self._best + eps:
            self._best = float(objective_cov)
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self._no_improve >= self._plateau_window:
                self._plateaued.add(target)
                self._completed.add(target)
                self._advance_after_completion_or_plateau()

        return self.status(objective_target=target, objective_cov=float(objective_cov))

    def _advance_after_completion_or_plateau(self) -> None:
        # Move to next non-completed target.
        self._index += 1
        self._best = 0.0
        self._no_improve = 0
        while 0 <= self._index < len(self._sequence) and self._sequence[self._index] in self._completed:
            self._index += 1

    def status(self, *, objective_target: Optional[str], objective_cov: float) -> GatingStatus:
        return GatingStatus(
            objective_target=objective_target,
            objective_coverage_fraction=float(objective_cov),
            gated_sequence=list(self._sequence),
            index=int(self._index),
            completed=set(self._completed),
            plateaued=set(self._plateaued),
        )


def compute_gating_sequence(hierarchy: Dict[str, Dict[str, object]], targets: List[str]) -> List[str]:
    """Compute a parent-first class order from an internal hierarchy map.

    - Only internal classes (keys of `hierarchy`) participate.
    - If `targets` is non-empty, include each target's internal ancestors.
    - Parents appear before children.
    """

    internal: Set[str] = set(hierarchy.keys())
    if not internal:
        return []

    # Choose seed nodes.
    if targets:
        seeds = [t for t in targets if t in internal]
    else:
        seeds = list(internal)

    if not seeds:
        return []

    # Expand to internal ancestors.
    selected: Set[str] = set()
    for s in seeds:
        cur = s
        while cur and cur in internal and cur not in selected:
            selected.add(cur)
            parent = hierarchy.get(cur, {}).get("parent_type_id")
            if isinstance(parent, str) and parent in internal:
                cur = parent
            else:
                break

    parent_of: Dict[str, Optional[str]] = {}
    for c in selected:
        p = hierarchy.get(c, {}).get("parent_type_id")
        parent_of[c] = str(p) if isinstance(p, str) and p in selected else None

    # Parent-first DFS order.
    order: List[str] = []
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(n: str) -> None:
        if n in visited:
            return
        if n in visiting:
            # Cycle: break deterministically by ignoring further recursion.
            return
        visiting.add(n)
        p = parent_of.get(n)
        if p:
            visit(p)
        visiting.remove(n)
        visited.add(n)
        order.append(n)

    for n in sorted(selected):
        visit(n)

    # Deduplicate preserving first occurrence.
    seen: Set[str] = set()
    deduped: List[str] = []
    for n in order:
        if n not in seen:
            seen.add(n)
            deduped.append(n)

    return deduped


def _coverage_fraction_or_none(summary: Optional[CoverageSummary]) -> Optional[float]:
    if summary is None:
        return 0.0
    return summary.coverage_fraction


def encode_method_objective(
    *,
    receiver_type_id: str,
    declaring_type_id: str,
    method_name: str,
    jvm_descriptor: str,
) -> str:
    # Descriptor does not contain '|', so this is safe and reversible.
    return f"{receiver_type_id}|{declaring_type_id}|{method_name}|{jvm_descriptor}"


def decode_method_objective(objective_id: str) -> Dict[str, str]:
    parts = objective_id.split("|", 3)
    if len(parts) != 4:
        return {
            "receiver_type_id": "",
            "declaring_type_id": "",
            "method_name": "",
            "jvm_descriptor": "",
        }
    return {
        "receiver_type_id": parts[0],
        "declaring_type_id": parts[1],
        "method_name": parts[2],
        "jvm_descriptor": parts[3],
    }


class MethodFirstGating:
    """Method-level gating (one method objective at a time).

    Objective identity is (receiver, declaring, method_name, jvm_descriptor).
    - receiver: the class under test (child context)
    - declaring: where the bytecode lives (JaCoCo attribution)

    We currently filter to public methods only, matching the generator's
    reflection strategy (`clazz.getMethods()`).
    """

    def __init__(self, *, completion_threshold: float = 0.99, plateau_window: int = 50):
        self._completion_threshold = float(completion_threshold)
        self._plateau_window = max(1, int(plateau_window))

        self._sequence: List[str] = []
        self._index: int = 0

        self._best: float = 0.0
        self._no_improve: int = 0

        self._completed: Set[str] = set()
        self._plateaued: Set[str] = set()

        self._project_key: Optional[Tuple[str, Tuple[str, ...]]] = None
        self._meta: Dict[str, Dict[str, str]] = {}

    def reset(self) -> None:
        self._sequence = []
        self._index = 0
        self._best = 0.0
        self._no_improve = 0
        self._completed = set()
        self._plateaued = set()
        self._project_key = None
        self._meta = {}

    def ensure_initialized(self, *, project: ProjectUnderTest, targets: List[str]) -> None:
        key = (str(project.root_path), tuple(sorted(targets)))
        if self._project_key == key and self._sequence:
            return

        self.reset()
        self._project_key = key
        self._sequence, self._meta = compute_method_gating_sequence(project=project, targets=targets)
        self._index = 0

    def current_objective_id(self) -> Optional[str]:
        while 0 <= self._index < len(self._sequence) and self._sequence[self._index] in self._completed:
            self._index += 1
            self._best = 0.0
            self._no_improve = 0
        if 0 <= self._index < len(self._sequence):
            return self._sequence[self._index]
        return None

    def current_objective(self) -> Optional[Dict[str, str]]:
        oid = self.current_objective_id()
        if oid is None:
            return None
        base = self._meta.get(oid) or decode_method_objective(oid)
        return dict(base) | {"objective_id": oid}

    def observe(self, *, objective_id: Optional[str], coverage: Optional[CoverageSummary]) -> GatingStatus:
        objective_cov = _coverage_fraction_or_none(coverage)
        if objective_id is None:
            return self.status(objective_target=None, objective_cov=1.0)

        if objective_cov is None:
            self._completed.add(objective_id)
            self._advance_after_completion_or_plateau()
            return self.status(objective_target=self.current_objective_id(), objective_cov=1.0)

        if objective_cov >= self._completion_threshold:
            self._completed.add(objective_id)
            self._advance_after_completion_or_plateau()
            return self.status(objective_target=self.current_objective_id(), objective_cov=float(objective_cov))

        eps = 1e-12
        if objective_cov > self._best + eps:
            self._best = float(objective_cov)
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self._no_improve >= self._plateau_window:
                self._plateaued.add(objective_id)
                self._completed.add(objective_id)
                self._advance_after_completion_or_plateau()

        return self.status(objective_target=objective_id, objective_cov=float(objective_cov))

    def _advance_after_completion_or_plateau(self) -> None:
        self._index += 1
        self._best = 0.0
        self._no_improve = 0
        while 0 <= self._index < len(self._sequence) and self._sequence[self._index] in self._completed:
            self._index += 1

    def status(self, *, objective_target: Optional[str], objective_cov: float) -> GatingStatus:
        return GatingStatus(
            objective_target=objective_target,
            objective_coverage_fraction=float(objective_cov),
            gated_sequence=list(self._sequence),
            index=int(self._index),
            completed=set(self._completed),
            plateaued=set(self._plateaued),
        )


def compute_method_gating_sequence(*, project: ProjectUnderTest, targets: List[str]) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """Compute a deterministic objective sequence over methods.

    Ordering:
    1) receiver classes in parent-first order (same as class gating)
    2) within each receiver: methods sorted by (declaring, name, descriptor)

    Returns:
    - sequence of objective_id strings
    - metadata map objective_id -> parts
    """

    receiver_order = compute_gating_sequence(project.hierarchy, targets)
    if not receiver_order:
        receiver_order = list(targets)

    callables = list(project.callable_methods or [])
    by_receiver: Dict[str, List[Dict[str, str]]] = {}
    for m in callables:
        if not isinstance(m, dict):
            continue
        receiver = m.get("receiver_type_id")
        declaring = m.get("declaring_type_id")
        name = m.get("method_name")
        desc = m.get("jvm_descriptor")
        vis = m.get("visibility")
        if not all(isinstance(x, str) and x for x in [receiver, declaring, name, desc]):
            continue
        if vis != "public":
            continue
        by_receiver.setdefault(receiver, []).append(
            {
                "receiver_type_id": str(receiver),
                "declaring_type_id": str(declaring),
                "method_name": str(name),
                "jvm_descriptor": str(desc),
            }
        )

    sequence: List[str] = []
    meta: Dict[str, Dict[str, str]] = {}
    for r in receiver_order:
        methods = by_receiver.get(r) or []
        methods_sorted = sorted(methods, key=lambda d: (d["declaring_type_id"], d["method_name"], d["jvm_descriptor"]))
        for md in methods_sorted:
            oid = encode_method_objective(
                receiver_type_id=md["receiver_type_id"],
                declaring_type_id=md["declaring_type_id"],
                method_name=md["method_name"],
                jvm_descriptor=md["jvm_descriptor"],
            )
            sequence.append(oid)
            meta[oid] = dict(md)

    return sequence, meta
