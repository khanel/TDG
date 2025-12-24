from __future__ import annotations

from pathlib import Path

import pytest

from RLOrchestrator.sbst.execution import locate_jacoco_xml
from RLOrchestrator.sbst.execution.runner import build_gradle_command


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_locate_jacoco_maven_default(tmp_path: Path):
    root = tmp_path / "sut"
    jacoco = root / "target" / "site" / "jacoco" / "jacoco.xml"
    _write(jacoco, "<report/>")

    found = locate_jacoco_xml(root, build_tool="maven")
    assert found == jacoco


def test_locate_jacoco_gradle_default(tmp_path: Path):
    root = tmp_path / "sut"
    jacoco = root / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml"
    _write(jacoco, "<report/>")

    found = locate_jacoco_xml(root, build_tool="gradle")
    assert found == jacoco


def test_locate_jacoco_override_relative(tmp_path: Path):
    root = tmp_path / "sut"
    override = root / "custom" / "jacoco.xml"
    _write(override, "<report/>")

    found = locate_jacoco_xml(root, build_tool="maven", override_path="custom/jacoco.xml")
    assert found == override


def test_gradle_wrapper_preferred_when_present(tmp_path: Path):
    root = tmp_path / "sut"
    root.mkdir()
    # Wrapper file exists -> should be used
    (root / "gradlew").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    cmd = build_gradle_command(root, tasks=["test"], use_wrapper=True)
    assert cmd[0].endswith("/gradlew")
