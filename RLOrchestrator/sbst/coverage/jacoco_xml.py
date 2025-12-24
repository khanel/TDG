from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

from .models import CoverageReport, CoverageSummary


class CoverageParseError(RuntimeError):
    pass


def method_key(declaring_fqn: str, method_name: str, jvm_descriptor: str) -> str:
    """Stable key for a method coverage entry.

    Format is intentionally human-readable and unambiguous.
    """

    return f"{declaring_fqn}::{method_name}{jvm_descriptor}"


def parse_jacoco_xml(path: str | Path) -> CoverageReport:
    """Parse JaCoCo XML and extract BRANCH counters.

    JaCoCo counter shape:
      <counter type="BRANCH" missed=".." covered=".."/>

    We provide:
    - overall branch counters (prefer report-level counter if present; else sum of classes)
    - per-class counters keyed by FQN (slashes converted to dots)
    """

    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        raise CoverageParseError(f"Unable to read JaCoCo XML: {p} ({exc})") from exc

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise CoverageParseError(f"Malformed JaCoCo XML: {p} ({exc})") from exc

    # Per-class and per-method counters
    by_class: Dict[str, CoverageSummary] = {}
    by_method: Dict[str, CoverageSummary] = {}
    for class_el in root.findall(".//package/class"):
        name = class_el.attrib.get("name")
        if not name:
            continue
        fqn = name.replace("/", ".")
        summary = _extract_branch_counter(class_el)
        if summary is None:
            continue
        by_class[fqn] = summary

        for method_el in class_el.findall("method"):
            mname = method_el.attrib.get("name")
            desc = method_el.attrib.get("desc")
            if not mname or not desc:
                continue
            msum = _extract_branch_counter(method_el)
            if msum is None:
                continue
            by_method[method_key(fqn, mname, desc)] = msum

    overall = _extract_branch_counter(root)
    if overall is None:
        # Fallback: sum over classes.
        overall = _sum_summaries(by_class)

    return CoverageReport(overall=overall, by_class=by_class, by_method=by_method)


def _extract_branch_counter(el: ET.Element) -> Optional[CoverageSummary]:
    for counter in el.findall("counter"):
        if counter.attrib.get("type") != "BRANCH":
            continue
        missed, covered = _parse_counter(counter)
        return CoverageSummary(branches_covered=covered, branches_missed=missed)
    return None


def _parse_counter(counter_el: ET.Element) -> Tuple[int, int]:
    try:
        missed = int(counter_el.attrib.get("missed", "0"))
        covered = int(counter_el.attrib.get("covered", "0"))
    except ValueError as exc:
        raise CoverageParseError(f"Invalid JaCoCo counter values: {counter_el.attrib}") from exc
    return missed, covered


def _sum_summaries(by_class: Dict[str, CoverageSummary]) -> CoverageSummary:
    missed = sum(s.branches_missed for s in by_class.values())
    covered = sum(s.branches_covered for s in by_class.values())
    return CoverageSummary(branches_covered=int(covered), branches_missed=int(missed))
