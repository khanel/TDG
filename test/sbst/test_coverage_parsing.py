from __future__ import annotations

from pathlib import Path

import pytest

from RLOrchestrator.sbst.coverage import CoverageParseError, method_key, parse_jacoco_xml


def test_parse_jacoco_overall_and_class_branch_counters(tmp_path: Path):
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <report name=\"demo\">
      <package name=\"com/acme\">
        <class name=\"com/acme/Foo\" sourcefilename=\"Foo.java\">
          <counter type=\"BRANCH\" missed=\"3\" covered=\"7\"/>
        </class>
        <class name=\"com/acme/Bar\" sourcefilename=\"Bar.java\">
          <counter type=\"BRANCH\" missed=\"0\" covered=\"0\"/>
        </class>
      </package>
      <counter type=\"BRANCH\" missed=\"5\" covered=\"15\"/>
    </report>
    """.strip()

    path = tmp_path / "jacoco.xml"
    path.write_text(xml, encoding="utf-8")

    report = parse_jacoco_xml(path)
    assert report.overall.branches_missed == 5
    assert report.overall.branches_covered == 15
    assert report.overall.coverage_fraction == 15 / 20

    foo = report.by_class["com.acme.Foo"]
    assert foo.branches_covered == 7
    assert foo.branches_missed == 3
    assert foo.coverage_fraction == 7 / 10


def test_parse_jacoco_fallback_overall_sum_when_missing_report_counter(tmp_path: Path):
    xml = """<report>
      <package name=\"x\">
        <class name=\"x/A\"><counter type=\"BRANCH\" missed=\"1\" covered=\"2\"/></class>
        <class name=\"x/B\"><counter type=\"BRANCH\" missed=\"3\" covered=\"4\"/></class>
      </package>
    </report>"""
    path = tmp_path / "jacoco.xml"
    path.write_text(xml, encoding="utf-8")

    report = parse_jacoco_xml(path)
    assert report.overall.branches_missed == 4
    assert report.overall.branches_covered == 6


def test_parse_jacoco_malformed_raises(tmp_path: Path):
    path = tmp_path / "jacoco.xml"
    path.write_text("<report><not-closed>", encoding="utf-8")
    with pytest.raises(CoverageParseError):
        parse_jacoco_xml(path)


def test_parse_jacoco_method_branch_counters(tmp_path: Path):
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <report name=\"demo\">
      <package name=\"com/acme\">
        <class name=\"com/acme/Foo\" sourcefilename=\"Foo.java\">
          <method name=\"foo\" desc=\"(II)I\" line=\"10\">
            <counter type=\"BRANCH\" missed=\"1\" covered=\"2\"/>
          </method>
          <method name=\"bar\" desc=\"()V\" line=\"20\">
            <counter type=\"BRANCH\" missed=\"0\" covered=\"0\"/>
          </method>
          <counter type=\"BRANCH\" missed=\"3\" covered=\"7\"/>
        </class>
      </package>
    </report>
    """.strip()

    path = tmp_path / "jacoco.xml"
    path.write_text(xml, encoding="utf-8")

    report = parse_jacoco_xml(path)
    k = method_key("com.acme.Foo", "foo", "(II)I")
    assert k in report.by_method
    assert report.by_method[k].branches_missed == 1
    assert report.by_method[k].branches_covered == 2
    assert report.by_method[k].coverage_fraction == 2 / 3
