from __future__ import annotations

from RLOrchestrator.sbst.pipeline.gating import ParentFirstGating, compute_gating_sequence
from RLOrchestrator.sbst.coverage.models import CoverageSummary
from RLOrchestrator.sbst.discovery.models import ProjectUnderTest


def test_compute_gating_sequence_parents_before_children():
    hierarchy = {
        "com.acme.Parent": {"parent_type_id": None},
        "com.acme.Child": {"parent_type_id": "com.acme.Parent"},
        "com.acme.GrandChild": {"parent_type_id": "com.acme.Child"},
    }

    seq = compute_gating_sequence(hierarchy, targets=["com.acme.GrandChild"])
    assert seq.index("com.acme.Parent") < seq.index("com.acme.Child") < seq.index("com.acme.GrandChild")


def test_gating_advances_on_completion():
    project = ProjectUnderTest(
        root_path="/tmp/sut",
        build_tool="maven",
        source_roots=[],
        test_roots=[],
        target_classes=["com.acme.Parent", "com.acme.Child"],
        hierarchy={
            "com.acme.Parent": {"parent_type_id": None, "children": ["com.acme.Child"], "is_external_parent": False, "type_id": "com.acme.Parent"},
            "com.acme.Child": {"parent_type_id": "com.acme.Parent", "children": [], "is_external_parent": False, "type_id": "com.acme.Child"},
        },
    )

    gating = ParentFirstGating(completion_threshold=0.99, plateau_window=3)
    gating.ensure_initialized(project=project, targets=["com.acme.Child"])

    cur = gating.current_target()
    assert cur == "com.acme.Parent"

    gating.observe(target=cur, coverage=CoverageSummary(branches_covered=99, branches_missed=0))
    assert gating.current_target() == "com.acme.Child"


def test_gating_plateaus_and_advances():
    project = ProjectUnderTest(
        root_path="/tmp/sut",
        build_tool="maven",
        source_roots=[],
        test_roots=[],
        target_classes=["A", "B"],
        hierarchy={
            "A": {"parent_type_id": None, "children": ["B"], "is_external_parent": False, "type_id": "A"},
            "B": {"parent_type_id": "A", "children": [], "is_external_parent": False, "type_id": "B"},
        },
    )

    gating = ParentFirstGating(completion_threshold=0.99, plateau_window=2)
    gating.ensure_initialized(project=project, targets=["B"])

    assert gating.current_target() == "A"
    # First observation establishes best; next two fail to improve -> plateau -> advance
    gating.observe(target="A", coverage=CoverageSummary(branches_covered=1, branches_missed=9))
    gating.observe(target="A", coverage=CoverageSummary(branches_covered=1, branches_missed=9))
    gating.observe(target="A", coverage=CoverageSummary(branches_covered=1, branches_missed=9))
    assert gating.current_target() == "B"
