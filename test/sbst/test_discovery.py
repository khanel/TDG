from __future__ import annotations

from pathlib import Path

import pytest

from RLOrchestrator.sbst.discovery import DiscoveryError, discover_project


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_detect_maven_project(tmp_path: Path):
    root = tmp_path / "sut"
    root.mkdir()
    _write(root / "pom.xml", "<project></project>")
    _write(
        root / "src" / "main" / "java" / "com" / "acme" / "Foo.java",
        """
        package com.acme;
        public class Foo { }
        """.strip(),
    )

    proj = discover_project(root)
    assert proj.build_tool == "maven"
    assert "com.acme.Foo" in proj.target_classes


def test_detect_gradle_project(tmp_path: Path):
    root = tmp_path / "sut"
    root.mkdir()
    _write(root / "build.gradle", "plugins { id 'java' }")
    _write(
        root / "src" / "main" / "java" / "com" / "acme" / "Bar.java",
        """
        package com.acme;
        public class Bar { }
        """.strip(),
    )

    proj = discover_project(root)
    assert proj.build_tool == "gradle"
    assert "com.acme.Bar" in proj.target_classes


def test_ambiguous_build_requires_override(tmp_path: Path):
    root = tmp_path / "sut"
    root.mkdir()
    _write(root / "pom.xml", "<project></project>")
    _write(root / "build.gradle", "plugins { id 'java' }")

    with pytest.raises(DiscoveryError):
        discover_project(root, build_tool_override="auto")

    proj = discover_project(root, build_tool_override="maven")
    assert proj.build_tool == "maven"


def test_extract_extends_relationship(tmp_path: Path):
    root = tmp_path / "sut"
    root.mkdir()
    _write(root / "pom.xml", "<project></project>")

    _write(
        root / "src" / "main" / "java" / "com" / "acme" / "Parent.java",
        """
        package com.acme;
        public class Parent { }
        """.strip(),
    )
    _write(
        root / "src" / "main" / "java" / "com" / "acme" / "Child.java",
        """
        package com.acme;
        public class Child extends Parent { }
        """.strip(),
    )

    proj = discover_project(root)
    node = proj.hierarchy.get("com.acme.Child")
    assert node is not None
    assert node["parent_type_id"] == "com.acme.Parent"
    assert node["is_external_parent"] is False
