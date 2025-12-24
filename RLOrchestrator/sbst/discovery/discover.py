from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import BuildTool, HierarchyNode, ProjectUnderTest, TypeId


_BUILD_FILES_MAVEN = ("pom.xml",)
_BUILD_FILES_GRADLE = ("build.gradle", "build.gradle.kts")

_EXCLUDED_DIR_NAMES = {"target", "build", "out", ".gradle", ".idea", ".git"}

_PACKAGE_RE = re.compile(r"^\s*package\s+([\w\.]+)\s*;", re.MULTILINE)
# MVP extraction: first declared type in file
_TYPE_RE = re.compile(
    r"^\s*(?:public\s+)?(?:abstract\s+)?(?:final\s+)?(class|interface|enum)\s+(\w+)\b",
    re.MULTILINE,
)
_EXTENDS_RE = re.compile(r"\bextends\s+([\w\.]+)")

# MVP method extraction from Java source.
# Intentionally conservative: extracts only methods with bodies (`{`) and skips
# private/static methods. This is meant to support method-level objective identity
# and parent->child seeding decisions; it is not a full Java parser.
_METHOD_RE = re.compile(
    r"^\s*(?P<mods>(?:public|protected|private|static|final|native|synchronized|abstract|strictfp|\s)+)?"
    r"\s*(?P<ret>[\w\.<>,\[\]]+)\s+"
    r"(?P<name>\w+)\s*\((?P<params>[^\)]*)\)\s*"
    r"(?:throws\s+[^\{]+)?\{",
    re.MULTILINE,
)

_PRIMITIVE_TO_DESC = {
    "boolean": "Z",
    "byte": "B",
    "char": "C",
    "short": "S",
    "int": "I",
    "long": "J",
    "float": "F",
    "double": "D",
    "void": "V",
}


class DiscoveryError(RuntimeError):
    pass


def discover_project(
    project_root: str | Path,
    *,
    build_tool_override: str = "auto",
    targets: Optional[Sequence[str]] = None,
) -> ProjectUnderTest:
    root = _find_project_root(Path(project_root))
    build_tool = _detect_build_tool(root, override=build_tool_override)

    source_roots, test_roots = _default_roots(root)

    discovered_types = _discover_types(source_roots)
    internal_fqns = {t[0] for t in discovered_types}  # fqn

    hierarchy = _build_hierarchy(discovered_types, internal_fqns)

    callable_methods = _discover_callable_methods(discovered_types, internal_fqns)

    if targets and len(list(targets)) > 0:
        target_classes = list(targets)
    else:
        target_classes = sorted(internal_fqns)

    return ProjectUnderTest(
        root_path=str(root),
        build_tool=build_tool,
        source_roots=[str(p) for p in source_roots],
        test_roots=[str(p) for p in test_roots],
        target_classes=target_classes,
        hierarchy={k: v for k, v in hierarchy.items()},
        callable_methods=callable_methods,
    )


def resolve_project_root(project_root: str | Path) -> Path:
    """Resolve the build root (directory containing pom.xml or build.gradle*)."""
    return _find_project_root(Path(project_root))


def _find_project_root(start: Path) -> Path:
    p = start
    if p.is_file():
        p = p.parent

    # Walk upwards until we find *some* recognized build file.
    for cur in [p] + list(p.parents):
        has_maven = any((cur / f).exists() for f in _BUILD_FILES_MAVEN)
        has_gradle = any((cur / f).exists() for f in _BUILD_FILES_GRADLE)
        if has_maven or has_gradle:
            return cur
    raise DiscoveryError(f"No Maven/Gradle build file found at/above: {start}")


def _detect_build_tool(root: Path, *, override: str) -> BuildTool:
    has_maven = any((root / f).exists() for f in _BUILD_FILES_MAVEN)
    has_gradle = any((root / f).exists() for f in _BUILD_FILES_GRADLE)

    if override in ("maven", "gradle"):
        if override == "maven" and not has_maven:
            raise DiscoveryError(f"build_tool override 'maven' but {root}/pom.xml not found")
        if override == "gradle" and not has_gradle:
            raise DiscoveryError(f"build_tool override 'gradle' but {root}/build.gradle* not found")
        return override  # type: ignore[return-value]

    # auto
    if has_maven and has_gradle:
        raise DiscoveryError(
            f"Ambiguous build tool: both Maven and Gradle files exist in {root}. "
            "Set build_tool explicitly."
        )
    if has_maven:
        return "maven"
    if has_gradle:
        return "gradle"

    raise DiscoveryError(f"No build files found in resolved project root: {root}")


def _default_roots(root: Path) -> Tuple[List[Path], List[Path]]:
    source = root / "src" / "main" / "java"
    tests = root / "src" / "test" / "java"

    source_roots: List[Path] = [source] if source.exists() else []
    test_roots: List[Path] = [tests] if tests.exists() else []
    return source_roots, test_roots


def _discover_types(source_roots: Sequence[Path]) -> List[Tuple[str, Optional[str], str]]:
    """Return list of (fqn, parent_fqn_or_none, file_path)."""
    results: List[Tuple[str, Optional[str], str]] = []

    for root in source_roots:
        for path in _iter_java_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            package = _extract_package(text)
            type_name = _extract_first_type_name(text)
            if not type_name:
                continue

            fqn = f"{package}.{type_name}" if package else type_name
            parent = _extract_extends(text)
            # Normalize simple parent names to package-local FQN.
            if parent and "." not in parent and package:
                parent = f"{package}.{parent}"

            results.append((fqn, parent, str(path)))

    return results


def _iter_java_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return

    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            for child in cur.iterdir():
                if child.is_dir():
                    if child.name in _EXCLUDED_DIR_NAMES:
                        continue
                    stack.append(child)
                elif child.is_file() and child.suffix == ".java":
                    yield child
        except OSError:
            continue


def _extract_package(text: str) -> str:
    m = _PACKAGE_RE.search(text)
    return m.group(1) if m else ""


def _extract_first_type_name(text: str) -> Optional[str]:
    m = _TYPE_RE.search(text)
    return m.group(2) if m else None


def _extract_extends(text: str) -> Optional[str]:
    m = _EXTENDS_RE.search(text)
    return m.group(1) if m else None


def _discover_callable_methods(
    discovered: Sequence[Tuple[str, Optional[str], str]],
    internal_fqns: Set[str],
) -> List[Dict[str, object]]:
    """Build a per-receiver callable method catalog.

    Each entry includes both receiver type and declaring type so objectives can be
    defined as "execute method M on receiver R" while JaCoCo attribution remains
    tied to the declaring class bytecode.

    Output is a flat list of dicts to keep JSON stable.
    """

    # 1) Extract declared methods per internal class.
    declared: Dict[str, Dict[Tuple[str, str], Dict[str, object]]] = {}
    parent_of: Dict[str, Optional[str]] = {fqn: parent for fqn, parent, _p in discovered}

    for fqn, _parent, path_str in discovered:
        path = Path(path_str)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            declared[fqn] = {}
            continue

        package = _extract_package(text)
        simple_name = fqn.split(".")[-1]

        methods: Dict[Tuple[str, str], Dict[str, object]] = {}
        for md in _extract_declared_methods(text, package=package, declaring_fqn=fqn, declaring_simple=simple_name):
            key = (str(md["method_name"]), str(md["jvm_descriptor"]))
            methods[key] = md
        declared[fqn] = methods

    # 2) For each receiver, union in inherited methods (first definition up chain).
    out: List[Dict[str, object]] = []
    for receiver_fqn in internal_fqns:
        seen: Set[Tuple[str, str]] = set()

        for key, md in declared.get(receiver_fqn, {}).items():
            seen.add(key)
            out.append(
                {
                    "receiver_type_id": receiver_fqn,
                    "declaring_type_id": receiver_fqn,
                    "method_name": md["method_name"],
                    "jvm_descriptor": md["jvm_descriptor"],
                    "inherited": False,
                    "visibility": md.get("visibility", "package"),
                }
            )

        cur = parent_of.get(receiver_fqn)
        while cur and cur in internal_fqns:
            for key, md in declared.get(cur, {}).items():
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "receiver_type_id": receiver_fqn,
                        "declaring_type_id": cur,
                        "method_name": md["method_name"],
                        "jvm_descriptor": md["jvm_descriptor"],
                        "inherited": True,
                        "visibility": md.get("visibility", "package"),
                    }
                )
            cur = parent_of.get(cur)

    return out


def _extract_declared_methods(
    text: str,
    *,
    package: str,
    declaring_fqn: str,
    declaring_simple: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for m in _METHOD_RE.finditer(text):
        mods = (m.group("mods") or "").strip()
        mods_lower = f" {mods.lower()} "

        if " private " in mods_lower:
            continue
        if " static " in mods_lower:
            continue

        ret = (m.group("ret") or "").strip()
        name = (m.group("name") or "").strip()
        params = (m.group("params") or "").strip()

        # Defensive: avoid matching weird constructs.
        if not name or not ret:
            continue

        visibility = "package"
        if " public " in mods_lower:
            visibility = "public"
        elif " protected " in mods_lower:
            visibility = "protected"
        elif " private " in mods_lower:
            visibility = "private"

        param_types = _parse_param_types(params)
        desc = _to_method_descriptor(
            param_types,
            ret,
            current_package=package,
        )

        out.append(
            {
                "declaring_type_id": declaring_fqn,
                "method_name": name,
                "jvm_descriptor": desc,
                "visibility": visibility,
                "declaring_simple": declaring_simple,
            }
        )
    return out


def _parse_param_types(params: str) -> List[str]:
    if not params.strip():
        return []

    parts = [p.strip() for p in params.split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        # Drop annotations and common modifiers.
        tokens = [t for t in p.replace("...", "[]").split() if not t.startswith("@")]
        tokens = [t for t in tokens if t not in {"final"}]
        if not tokens:
            continue

        # Heuristic: last token is usually param name; type is everything before.
        # Examples:
        #   "int x" -> ["int", "x"]
        #   "java.lang.String s" -> ["java.lang.String", "s"]
        #   "String[] xs" -> ["String[]", "xs"]
        if len(tokens) >= 2:
            type_token = " ".join(tokens[:-1])
        else:
            type_token = tokens[0]

        out.append(type_token.strip())
    return out


def _to_method_descriptor(param_types: Sequence[str], return_type: str, *, current_package: str) -> str:
    args = "".join(_to_type_descriptor(t, current_package=current_package) for t in param_types)
    ret = _to_type_descriptor(return_type, current_package=current_package)
    return f"({args}){ret}"


def _to_type_descriptor(java_type: str, *, current_package: str) -> str:
    t = java_type.strip()
    # Strip generics e.g. List<String> -> List
    t = re.sub(r"<[^>]*>", "", t).strip()

    # Arrays
    dims = 0
    while t.endswith("[]"):
        dims += 1
        t = t[: -2].strip()

    base = _PRIMITIVE_TO_DESC.get(t)
    if base is None:
        # Common simple name resolution.
        if t == "String" or t.endswith(".String"):
            internal = "java/lang/String"
        elif "." in t:
            internal = t.replace(".", "/")
        elif current_package:
            internal = f"{current_package.replace('.', '/')}/{t}"
        else:
            internal = t
        base = f"L{internal};"

    return ("[" * dims) + base


def _build_hierarchy(
    discovered: Sequence[Tuple[str, Optional[str], str]],
    internal_fqns: Set[str],
) -> Dict[str, Dict[str, object]]:
    """Build a minimal hierarchy model keyed by child FQN.

    Output JSON shape is kept simple and stable for early debugging.
    """

    nodes: Dict[str, HierarchyNode] = {}

    for child_fqn, parent_fqn, _path in discovered:
        parent_type = TypeId(parent_fqn) if parent_fqn else None
        node = HierarchyNode(
            type_id=TypeId(child_fqn),
            parent_type_id=parent_type,
            is_external_parent=bool(parent_fqn and parent_fqn not in internal_fqns),
            children=[],
        )
        nodes[child_fqn] = node

    for child_fqn, node in nodes.items():
        if node.parent_type_id is None:
            continue
        parent_fqn = node.parent_type_id.fqn
        if parent_fqn in nodes:
            nodes[parent_fqn].children.append(TypeId(child_fqn))

    # Serialize
    out: Dict[str, Dict[str, object]] = {}
    for fqn, node in nodes.items():
        out[fqn] = {
            "type_id": node.type_id.fqn,
            "parent_type_id": node.parent_type_id.fqn if node.parent_type_id else None,
            "is_external_parent": bool(node.is_external_parent),
            "children": [c.fqn for c in node.children],
        }
    return out
