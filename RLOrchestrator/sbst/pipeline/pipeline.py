from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from Core.problem import Solution

from .artifacts import prepare_artifacts, write_json, write_text
from .models import SBSTConfig
from ..discovery import DiscoveryError, discover_project
from ..discovery import resolve_project_root
from ..execution import (
    ExecutionError,
    build_gradle_command,
    build_maven_command,
    locate_jacoco_xml,
    run_command,
)
from ..coverage import CoverageParseError, parse_jacoco_xml
from ..generation import (
    GenerationConfig,
    delete_paths,
    generate_junit5_tests,
    generated_tests_digest,
    write_tests_to_directory,
)
from ..cache import (
    build_fingerprint_snapshot,
    discovery_input_files,
    execution_input_files,
    load_entry,
    mtime_first_then_hash_unchanged,
    save_entry,
    stable_key,
)

from .gating import MethodFirstGating, ParentFirstGating


@dataclass(frozen=True)
class SBSTEvaluationResult:
    fitness: float
    coverage_fraction: Optional[float] = None
    error: Optional[str] = None

    # Objective metadata (best-effort). When gating is enabled this allows
    # objective-aware population seeding.
    objective_kind: Optional[str] = None  # "class" | "method"
    objective_token: Optional[str] = None  # class FQN or method objective_id
    next_objective_token: Optional[str] = None

    # Method objective details (when objective_kind == "method").
    objective_id: Optional[str] = None
    objective_method_key: Optional[str] = None
    objective_receiver_type_id: Optional[str] = None
    objective_declaring_type_id: Optional[str] = None
    objective_method_name: Optional[str] = None
    objective_jvm_descriptor: Optional[str] = None


class SBSTPipeline:
    """SBST evaluation pipeline.

    End-to-end evaluation:
    - normalize candidate representation (genes)
    - discover project + internal inheritance (cached)
    - generate deterministic JUnit 5 tests
    - execute Maven/Gradle and locate JaCoCo XML (cached)
    - parse branch coverage and compute fitness `f = 1 - c`
    - (optional) apply parent-first gating + plateau progression across targets

    Contract: never raise; always return a numeric fitness.
    """

    def __init__(self, config: SBSTConfig):
        self._cfg = config
        self._gating = ParentFirstGating(
            completion_threshold=float(config.gating_complete_threshold),
            plateau_window=int(config.plateau_window),
        )
        self._method_gating = MethodFirstGating(
            completion_threshold=float(config.gating_complete_threshold),
            plateau_window=int(config.plateau_window),
        )

    def evaluate(self, solution: Solution) -> SBSTEvaluationResult:
        candidate = _extract_candidate(solution, dimension=self._cfg.dimension)
        artifacts = prepare_artifacts(self._cfg.resolved_work_dir(), solution_id=getattr(solution, "id", None), candidate=candidate)

        # Always write resolved config and candidate for reproducibility/debuggability.
        write_json(artifacts.config_json, self._cfg.to_json_dict())
        write_json(artifacts.candidate_json, {"candidate": candidate, "solution_id": getattr(solution, "id", None)})

        # Stage-2: project discovery (best-effort). Never crash.
        project = None
        if self._cfg.project_root:
            try:
                # Discovery cache key depends on resolved inputs that affect discovery.
                disc_key = stable_key(
                    {
                        "kind": "discovery",
                        "project_root": str(self._cfg.project_root),
                        "build_tool": str(self._cfg.build_tool),
                    }
                )

                cached = load_entry(self._cfg.resolved_work_dir(), namespace="discovery", key=disc_key)
                resolved_root = resolve_project_root(self._cfg.project_root)
                assumed_source_roots = [resolved_root / "src" / "main" / "java"]
                disc_inputs = discovery_input_files(resolved_root, [p for p in assumed_source_roots if p.exists()])

                if cached is not None:
                    ok, new_snapshot = mtime_first_then_hash_unchanged(previous=cached.fingerprints, paths=disc_inputs)
                    if ok:
                        if new_snapshot != cached.fingerprints:
                            save_entry(
                                self._cfg.resolved_work_dir(),
                                namespace="discovery",
                                key=disc_key,
                                fingerprints=new_snapshot,
                                payload=cached.payload,
                            )

                        payload = dict(cached.payload)
                        payload["cached"] = True
                        write_json(artifacts.discovery_json, payload)
                        from ..discovery.models import ProjectUnderTest

                        project = ProjectUnderTest.from_json_dict(payload)
                    else:
                        project = discover_project(
                            self._cfg.project_root,
                            build_tool_override=self._cfg.build_tool,
                            targets=self._cfg.targets,
                        )
                        snapshot = build_fingerprint_snapshot(disc_inputs)
                        save_entry(
                            self._cfg.resolved_work_dir(),
                            namespace="discovery",
                            key=disc_key,
                            fingerprints=snapshot,
                            payload=project.to_json_dict(),
                        )
                        write_json(artifacts.discovery_json, project.to_json_dict() | {"cached": False})
                else:
                    project = discover_project(
                        self._cfg.project_root,
                        build_tool_override=self._cfg.build_tool,
                        targets=self._cfg.targets,
                    )
                    snapshot = build_fingerprint_snapshot(disc_inputs)
                    save_entry(
                        self._cfg.resolved_work_dir(),
                        namespace="discovery",
                        key=disc_key,
                        fingerprints=snapshot,
                        payload=project.to_json_dict(),
                    )
                    write_json(artifacts.discovery_json, project.to_json_dict() | {"cached": False})
            except DiscoveryError as exc:
                write_json(
                    artifacts.discovery_json,
                    {
                        "error": str(exc),
                        "project_root": self._cfg.project_root,
                        "build_tool": self._cfg.build_tool,
                        "targets": list(self._cfg.targets),
                    },
                )
        else:
            write_json(
                artifacts.discovery_json,
                {
                    "note": "No project_root configured; discovery skipped.",
                    "project_root": None,
                },
            )

        try:
            # Stage-3: run build tool + obtain JaCoCo XML when project_root is configured.
            if project is not None:
                root = Path(project.root_path)

                # Stage-7: determine current gated objective target.
                objective_kind = str(getattr(self._cfg, "objective_granularity", "class"))
                objective_target: Optional[str] = None
                objective_method: Optional[Dict[str, str]] = None
                if bool(self._cfg.gating_enabled) and objective_kind == "class":
                    self._gating.ensure_initialized(project=project, targets=list(self._cfg.targets))
                    objective_target = self._gating.current_target()
                elif bool(self._cfg.gating_enabled) and objective_kind == "method":
                    self._method_gating.ensure_initialized(project=project, targets=list(self._cfg.targets))
                    objective_method = self._method_gating.current_objective()

                # Stage-6: generate tests deterministically.
                gen_cfg = GenerationConfig(
                    max_tests_per_candidate=int(self._cfg.max_tests_per_candidate),
                    max_actions_per_test=int(self._cfg.max_actions_per_test),
                    package_strategy=str(self._cfg.package_strategy),
                    fixed_package=str(self._cfg.fixed_test_package),
                )
                generated = generate_junit5_tests(
                    candidate=candidate,
                    project=project,
                    candidate_digest=artifacts.candidate_digest,
                    targets=([objective_method["receiver_type_id"]] if objective_method else (list(self._cfg.targets) if self._cfg.targets else None)),
                    cfg=gen_cfg,
                )
                gen_digest = generated_tests_digest(generated)
                # Always store generated sources in the run artifacts.
                write_tests_to_directory(generated, artifacts.tests_dir)

                # Execution/coverage cache check.
                exec_key = stable_key(
                    {
                        "kind": "execution",
                        "project_root": str(project.root_path),
                        "build_tool": str(project.build_tool),
                        "targets": list(self._cfg.targets),
                        "objective_kind": objective_kind,
                        "objective_target": objective_target,
                        "objective_method": objective_method,
                        "candidate": candidate,
                        "generated_tests_digest": gen_digest,
                        "maven_goals": list(self._cfg.maven_goals),
                        "gradle_tasks": list(self._cfg.gradle_tasks),
                        "gradle_use_wrapper": bool(self._cfg.gradle_use_wrapper),
                        "jacoco_xml_path": self._cfg.jacoco_xml_path,
                        "gating_enabled": bool(self._cfg.gating_enabled),
                        "gating_complete_threshold": float(self._cfg.gating_complete_threshold),
                        "plateau_window": int(self._cfg.plateau_window),
                    }
                )

                inputs = execution_input_files(root, [Path(p) for p in project.source_roots])
                written: List[Path] = []
                cached_exec = load_entry(self._cfg.resolved_work_dir(), namespace="execution", key=exec_key)
                if cached_exec is not None:
                    ok, _snap = mtime_first_then_hash_unchanged(previous=cached_exec.fingerprints, paths=inputs)
                    if ok:
                        payload = dict(cached_exec.payload)
                        payload["cached"] = True
                        # Best-effort: copy cached jacoco xml into this run for inspection.
                        cached_jacoco = payload.get("cached_jacoco_xml")
                        if isinstance(cached_jacoco, str):
                            try:
                                src = Path(cached_jacoco)
                                if src.exists() and src.is_file():
                                    artifacts.jacoco_xml.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                            except OSError:
                                pass
                        # Best-effort: reuse execution metadata files for this run.
                        if "command" in payload:
                            write_text(artifacts.command_txt, str(payload.get("command")) + "\n")
                        if "stdout" in payload:
                            write_text(artifacts.stdout_txt, str(payload.get("stdout") or ""))
                        if "stderr" in payload:
                            write_text(artifacts.stderr_txt, str(payload.get("stderr") or ""))
                        if "exit_code_json" in payload and isinstance(payload.get("exit_code_json"), dict):
                            write_json(artifacts.exit_code_json, payload["exit_code_json"])

                        write_json(artifacts.coverage_summary_json, payload)

                        # Advance gating state using cached objective observation.
                        if bool(self._cfg.gating_enabled) and payload.get("objective_kind") == "class":
                            from ..coverage.models import CoverageSummary

                            tgt = payload.get("objective_target")
                            if isinstance(tgt, str) and tgt:
                                obj_total = payload.get("objective_total_branches")
                                obj_cov = payload.get("coverage_fraction")
                                if obj_total is not None and int(obj_total) <= 0:
                                    self._gating.observe(target=tgt, coverage=CoverageSummary(0, 0))
                                elif obj_cov is not None:
                                    denom = 1000
                                    covered = int(round(float(obj_cov) * denom))
                                    missed = max(0, denom - covered)
                                    self._gating.observe(target=tgt, coverage=CoverageSummary(covered, missed))
                        if bool(self._cfg.gating_enabled) and payload.get("objective_kind") == "method":
                            from ..coverage.models import CoverageSummary

                            oid = payload.get("objective_id")
                            if isinstance(oid, str) and oid:
                                obj_total = payload.get("objective_total_branches")
                                obj_cov = payload.get("coverage_fraction")
                                if obj_total is not None and int(obj_total) <= 0:
                                    self._method_gating.observe(objective_id=oid, coverage=CoverageSummary(0, 0))
                                elif obj_cov is not None:
                                    denom = 1000
                                    covered = int(round(float(obj_cov) * denom))
                                    missed = max(0, denom - covered)
                                    self._method_gating.observe(objective_id=oid, coverage=CoverageSummary(covered, missed))

                        fitness = float(payload.get("fitness", 1.0))
                        cov = payload.get("coverage_fraction")
                        objective_kind = payload.get("objective_kind")
                        objective_target = payload.get("objective_target")
                        objective_id = payload.get("objective_id")
                        objective_token = objective_id if objective_kind == "method" else objective_target
                        next_token = payload.get("next_objective_target")

                        return SBSTEvaluationResult(
                            fitness=fitness,
                            coverage_fraction=float(cov) if cov is not None else None,
                            objective_kind=str(objective_kind) if isinstance(objective_kind, str) else None,
                            objective_token=str(objective_token) if isinstance(objective_token, str) else None,
                            next_objective_token=str(next_token) if isinstance(next_token, str) else None,
                            objective_id=str(objective_id) if isinstance(objective_id, str) else None,
                            objective_method_key=str(payload.get("objective_method_key")) if isinstance(payload.get("objective_method_key"), str) else None,
                            objective_receiver_type_id=str(payload.get("objective_receiver_type_id")) if isinstance(payload.get("objective_receiver_type_id"), str) else None,
                            objective_declaring_type_id=str(payload.get("objective_declaring_type_id")) if isinstance(payload.get("objective_declaring_type_id"), str) else None,
                            objective_method_name=str(payload.get("objective_method_name")) if isinstance(payload.get("objective_method_name"), str) else None,
                            objective_jvm_descriptor=str(payload.get("objective_jvm_descriptor")) if isinstance(payload.get("objective_jvm_descriptor"), str) else None,
                        )

                # Write tests into the SUT test tree for this execution.
                test_root = Path(project.test_roots[0]) if project.test_roots else (root / "src" / "test" / "java")
                written = write_tests_to_directory(generated, test_root)

                # Build command
                if project.build_tool == "maven":
                    command = build_maven_command(goals=list(self._cfg.maven_goals))
                else:
                    command = build_gradle_command(
                        root,
                        tasks=list(self._cfg.gradle_tasks),
                        use_wrapper=bool(self._cfg.gradle_use_wrapper),
                    )

                write_text(artifacts.command_txt, " ".join(command) + "\n")
                try:
                    exec_result = run_command(command, cwd=root, timeout_seconds=int(self._cfg.timeout_seconds))
                finally:
                    # Keep the SUT clean: delete generated sources after run.
                    delete_paths(written)
                write_text(artifacts.stdout_txt, exec_result.stdout)
                write_text(artifacts.stderr_txt, exec_result.stderr)
                write_json(artifacts.exit_code_json, exec_result.to_json_dict())

                if not exec_result.ok:
                    write_json(
                        artifacts.coverage_summary_json,
                        {
                            "branches_covered": None,
                            "branches_missed": None,
                            "coverage_fraction": None,
                            "fitness": 1.0,
                            "error": "build_or_test_failed",
                            "timed_out": bool(exec_result.timed_out),
                            "exit_code": exec_result.exit_code,
                        },
                    )
                    return SBSTEvaluationResult(fitness=1.0, coverage_fraction=None, error="build_or_test_failed")

                jacoco_src = locate_jacoco_xml(
                    root,
                    build_tool=project.build_tool,
                    override_path=self._cfg.jacoco_xml_path,
                )
                if jacoco_src is None:
                    write_json(
                        artifacts.coverage_summary_json,
                        {
                            "branches_covered": None,
                            "branches_missed": None,
                            "coverage_fraction": None,
                            "fitness": 1.0,
                            "error": "jacoco_xml_missing",
                        },
                    )
                    return SBSTEvaluationResult(fitness=1.0, coverage_fraction=None, error="jacoco_xml_missing")

                # Copy XML into run artifacts for stable inspection.
                artifacts.jacoco_xml.write_text(jacoco_src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

                # Stage-4: parse JaCoCo and compute fitness from branch coverage.
                report = parse_jacoco_xml(artifacts.jacoco_xml)

                overall_cov = report.overall.coverage_fraction
                overall_cov_f = float(overall_cov) if overall_cov is not None else 1.0

                # Stage-7: objective selection and progression.
                objective_total_branches: Optional[int] = None
                next_objective_target = None
                objective_id: Optional[str] = None
                objective_method_key: Optional[str] = None

                if bool(self._cfg.gating_enabled) and objective_kind == "class" and objective_target is not None:
                    objective_summary = report.by_class.get(objective_target)
                    if objective_summary is None:
                        objective_cov_f = 0.0
                    else:
                        objective_total_branches = int(objective_summary.total_branches)
                        cov = objective_summary.coverage_fraction
                        objective_cov_f = float(cov) if cov is not None else 1.0
                    self._gating.observe(target=objective_target, coverage=objective_summary)
                    next_objective_target = self._gating.current_target()

                elif bool(self._cfg.gating_enabled) and objective_kind == "method" and objective_method is not None:
                    from ..coverage import method_key

                    objective_id = objective_method.get("objective_id")
                    declaring = objective_method.get("declaring_type_id", "")
                    mname = objective_method.get("method_name", "")
                    desc = objective_method.get("jvm_descriptor", "")
                    objective_method_key = method_key(str(declaring), str(mname), str(desc))

                    objective_summary = report.by_method.get(objective_method_key)
                    if objective_summary is None:
                        objective_cov_f = 0.0
                    else:
                        objective_total_branches = int(objective_summary.total_branches)
                        cov = objective_summary.coverage_fraction
                        objective_cov_f = float(cov) if cov is not None else 1.0

                    self._method_gating.observe(objective_id=objective_id, coverage=objective_summary)
                    nxt = self._method_gating.current_objective()
                    next_objective_target = nxt.get("objective_id") if nxt else None

                else:
                    objective_cov_f = float(_select_coverage_fraction(report=report, targets=list(self._cfg.targets)))

                fitness = 1.0 - float(objective_cov_f)

                write_json(
                    artifacts.coverage_summary_json,
                    {
                        "branches_covered": report.overall.branches_covered,
                        "branches_missed": report.overall.branches_missed,
                        "coverage_fraction": objective_cov_f,
                        "overall_coverage_fraction": overall_cov_f,
                        "objective_total_branches": objective_total_branches,
                        "fitness": fitness,
                        "jacoco_xml_source_path": str(jacoco_src),
                        "note": "Fitness computed from JaCoCo BRANCH coverage (f = 1 - c).",
                        "objective_kind": objective_kind,
                        "objective_target": objective_target,
                        "objective_id": objective_id,
                        "objective_method_key": objective_method_key,
                        "objective_receiver_type_id": objective_method.get("receiver_type_id") if objective_method else None,
                        "objective_declaring_type_id": objective_method.get("declaring_type_id") if objective_method else None,
                        "objective_method_name": objective_method.get("method_name") if objective_method else None,
                        "objective_jvm_descriptor": objective_method.get("jvm_descriptor") if objective_method else None,
                        "next_objective_target": next_objective_target,
                        "by_class": {
                            k: v.to_json_dict() for k, v in report.by_class.items()
                        },
                        "by_method": {
                            k: v.to_json_dict() for k, v in report.by_method.items()
                        },
                        "gating": {
                            "enabled": bool(self._cfg.gating_enabled),
                            "completion_threshold": float(self._cfg.gating_complete_threshold),
                            "plateau_window": int(self._cfg.plateau_window),
                        },
                    },
                )

                # Persist execution cache entry on success.
                snap2 = build_fingerprint_snapshot(inputs)
                save_entry(
                    self._cfg.resolved_work_dir(),
                    namespace="execution",
                    key=exec_key,
                    fingerprints=snap2,
                    payload={
                        "cached": False,
                        "fitness": fitness,
                        "coverage_fraction": objective_cov_f,
                        "overall_coverage_fraction": overall_cov_f,
                        "objective_total_branches": objective_total_branches,
                        "branches_covered": report.overall.branches_covered,
                        "branches_missed": report.overall.branches_missed,
                        "cached_jacoco_xml": str(artifacts.jacoco_xml),
                        "build_tool": str(project.build_tool),
                        "project_root": str(project.root_path),
                        "targets": list(self._cfg.targets),
                        "generated_tests_digest": gen_digest,
                        "objective_kind": objective_kind,
                        "objective_target": objective_target,
                        "objective_id": objective_id,
                        "objective_method_key": objective_method_key,
                        "objective_method": objective_method,
                        "next_objective_target": next_objective_target,
                        "command": " ".join(command),
                        "stdout": exec_result.stdout,
                        "stderr": exec_result.stderr,
                        "exit_code_json": exec_result.to_json_dict(),
                    },
                )
                objective_token = objective_id if objective_kind == "method" else objective_target
                next_token = next_objective_target
                return SBSTEvaluationResult(
                    fitness=fitness,
                    coverage_fraction=objective_cov_f,
                    objective_kind=str(objective_kind),
                    objective_token=str(objective_token) if isinstance(objective_token, str) else None,
                    next_objective_token=str(next_token) if isinstance(next_token, str) else None,
                    objective_id=objective_id,
                    objective_method_key=objective_method_key,
                    objective_receiver_type_id=objective_method.get("receiver_type_id") if objective_method else None,
                    objective_declaring_type_id=objective_method.get("declaring_type_id") if objective_method else None,
                    objective_method_name=objective_method.get("method_name") if objective_method else None,
                    objective_jvm_descriptor=objective_method.get("jvm_descriptor") if objective_method else None,
                )

            # No project configured: keep Stage-1 surrogate fitness.
            fitness = _surrogate_fitness(candidate)
            write_json(
                artifacts.coverage_summary_json,
                {
                    "branches_covered": None,
                    "branches_missed": None,
                    "coverage_fraction": None,
                    "fitness": fitness,
                    "note": "Stage-1 scaffold: surrogate fitness (no Java execution yet).",
                },
            )
            return SBSTEvaluationResult(fitness=fitness, coverage_fraction=None)

        except (ExecutionError, CoverageParseError, OSError) as exc:
            write_text(artifacts.stderr_txt, f"SBST execution error: {exc!r}\n")
            write_json(
                artifacts.coverage_summary_json,
                {
                    "branches_covered": None,
                    "branches_missed": None,
                    "coverage_fraction": None,
                    "fitness": 1.0,
                    "error": repr(exc),
                },
            )
            return SBSTEvaluationResult(fitness=1.0, coverage_fraction=None, error=repr(exc))

        except Exception as exc:  # noqa: BLE001 - must never crash solvers
            write_text(artifacts.stderr_txt, f"SBST evaluation error: {exc!r}\n")
            write_json(
                artifacts.coverage_summary_json,
                {
                    "branches_covered": None,
                    "branches_missed": None,
                    "coverage_fraction": None,
                    "fitness": 1.0,
                    "error": repr(exc),
                },
            )
            return SBSTEvaluationResult(fitness=1.0, coverage_fraction=None, error=repr(exc))


def _extract_candidate(solution: Solution, *, dimension: int) -> Dict[str, List[int]]:
    rep = getattr(solution, "representation", None)
    genes: List[int]

    if isinstance(rep, dict) and "genes" in rep:
        genes = list(rep.get("genes") or [])
    elif isinstance(rep, list):
        genes = list(rep)
    else:
        genes = []

    if not genes:
        genes = [0] * max(1, int(dimension))

    # Normalize type to Python ints.
    genes = [int(x) for x in genes[: max(1, int(dimension))]]
    if len(genes) < max(1, int(dimension)):
        genes = genes + [0] * (max(1, int(dimension)) - len(genes))

    return {"genes": genes}


def _surrogate_fitness(candidate: Dict[str, List[int]]) -> float:
    # Surrogate objective in [0, 1]: mean squared distance from 0.5.
    genes = candidate.get("genes") or []
    arr = np.asarray(genes, dtype=float)
    if arr.size == 0:
        return 1.0

    scaled = (arr % 10) / 9.0  # map ints to [0,1]
    fitness = float(np.clip(np.mean((scaled - 0.5) ** 2) * 4.0, 0.0, 1.0))
    return fitness


def _select_coverage_fraction(*, report, targets: List[str]) -> float:
    # If explicit targets are provided, aggregate their class-level counters.
    if targets:
        covered = 0
        missed = 0
        for t in targets:
            s = report.by_class.get(t)
            if s is None:
                continue
            if s.total_branches <= 0:
                # Exclude no-branch classes from objectives (master plan rule).
                continue
            covered += int(s.branches_covered)
            missed += int(s.branches_missed)

        denom = covered + missed
        if denom <= 0:
            # Nothing meaningful to cover under the target set.
            return 1.0
        return float(covered) / float(denom)

    # Default: use overall report coverage when no explicit targets.
    c = report.overall.coverage_fraction
    return float(c) if c is not None else 1.0
