#!/usr/bin/env python3
"""Stage-10 validation: discovery works on the in-repo demo SUT.

This test is intentionally lightweight and does not require Maven/Java.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RLOrchestrator.sbst.discovery import discover_project


def test_demo_sut_discovery_inheritance_and_targets():
    project_root = Path(__file__).parent.parent.parent / "examples" / "sbst-demo-java-maven"
    assert project_root.exists()

    targets = [
        "com.example.sbstdemo.BaseLogic",
        "com.example.sbstdemo.ChildLogic",
        "com.example.sbstdemo.GrandChildLogic",
    ]

    proj = discover_project(str(project_root), build_tool_override="maven", targets=targets)

    assert proj.build_tool == "maven"

    for t in targets:
        assert t in proj.target_classes

    # Internal inheritance chain should be present.
    assert proj.hierarchy.get("com.example.sbstdemo.ChildLogic", {}).get("parent_type_id") == "com.example.sbstdemo.BaseLogic"
    assert proj.hierarchy.get("com.example.sbstdemo.GrandChildLogic", {}).get("parent_type_id") == "com.example.sbstdemo.ChildLogic"

    # Callable methods should include inherited parent methods for child receivers.
    # Objective identity is (receiver, declaring, name, descriptor).
    methods = proj.callable_methods
    assert isinstance(methods, list)

    def has(receiver: str, declaring: str, name: str, desc: str, inherited: bool) -> bool:
        for m in methods:
            if (
                m.get("receiver_type_id") == receiver
                and m.get("declaring_type_id") == declaring
                and m.get("method_name") == name
                and m.get("jvm_descriptor") == desc
                and bool(m.get("inherited")) is bool(inherited)
            ):
                return True
        return False

    # Parent declared method callable on parent.
    assert has(
        "com.example.sbstdemo.BaseLogic",
        "com.example.sbstdemo.BaseLogic",
        "baseCompare",
        "(II)I",
        False,
    )

    # Same parent method should be callable on ChildLogic receiver, but attributed to BaseLogic.
    assert has(
        "com.example.sbstdemo.ChildLogic",
        "com.example.sbstdemo.BaseLogic",
        "baseCompare",
        "(II)I",
        True,
    )

    # Child declared method should be attributed to child.
    assert has(
        "com.example.sbstdemo.ChildLogic",
        "com.example.sbstdemo.ChildLogic",
        "childCompare",
        "(II)I",
        False,
    )
