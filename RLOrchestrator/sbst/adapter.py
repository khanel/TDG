"""SBST TDG adapter (Stage-1 contract).

This adapter is intentionally usable *today* by the orchestrator/solvers, while also
locking in the SBST evaluation contract we need for the real Java pipeline:

- A stable candidate representation (currently: integer genes)
- A stable evaluation entrypoint that never crashes and always returns a fitness
- Standard run/artifact directories for debugging
- A configuration surface that can be overridden via the registry

SBST evaluation is implemented by `SBSTPipeline` (generation + Maven/Gradle +
JaCoCo parsing). The adapter must never crash and must always return a numeric
fitness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Core.problem import ProblemInterface, Solution

from .pipeline.models import SBSTConfig
from .pipeline.pipeline import SBSTPipeline


_ELITE_PER_OBJECTIVE = 8


class SBSTAdapter(ProblemInterface):
    """SBST adapter.

    Representation (current scaffold):
    - `solution.representation` is a dict with a `genes: list[int]` field.

        Fitness:
        - Minimizes `1 - branch_coverage(objective_target)` where branch coverage is
            measured from JaCoCo XML. Objective targeting may be gated (parent-first).
    """

    def __init__(
        self,
        *,
        dimension: int = 24,
        seed: Optional[int] = 42,
        # Stage-1 config overrides (future stages will rely on these)
        project_root: Optional[str] = None,
        build_tool: str = "auto",
        targets: Optional[List[str]] = None,
        work_dir: str = "runs/sbst",
        timeout_seconds: int = 300,
        gradle_use_wrapper: bool = True,
        maven_goals: Optional[List[str]] = None,
        gradle_tasks: Optional[List[str]] = None,
        jacoco_xml_path: Optional[str] = None,
        max_tests_per_candidate: int = 1,
        max_actions_per_test: int = 5,
        package_strategy: str = "match_target",
        fixed_test_package: str = "generated.sbst",
        gating_enabled: bool = True,
        gating_complete_threshold: float = 0.99,
        plateau_window: int = 50,
        objective_granularity: str = "class",
    ):
        self._cfg = SBSTConfig(
            dimension=max(1, int(dimension)),
            seed=seed,
            project_root=project_root,
            build_tool=build_tool if build_tool in ("auto", "maven", "gradle") else "auto",
            targets=list(targets or []),
            work_dir=str(work_dir),
            timeout_seconds=int(timeout_seconds),
            gradle_use_wrapper=bool(gradle_use_wrapper),
            maven_goals=list(maven_goals) if maven_goals is not None else ["test", "jacoco:report"],
            gradle_tasks=list(gradle_tasks) if gradle_tasks is not None else ["test", "jacocoTestReport"],
            jacoco_xml_path=jacoco_xml_path,
            max_tests_per_candidate=int(max_tests_per_candidate),
            max_actions_per_test=int(max_actions_per_test),
            package_strategy=str(package_strategy),
            fixed_test_package=str(fixed_test_package),
            gating_enabled=bool(gating_enabled),
            gating_complete_threshold=float(gating_complete_threshold),
            plateau_window=int(plateau_window),
            objective_granularity=str(objective_granularity) if str(objective_granularity) in {"class", "method"} else "class",
        )
        self._rng = np.random.default_rng(seed)
        self._bounds = {"lower_bound": 0.0, "upper_bound": 1.0}
        self._pipeline = SBSTPipeline(self._cfg)

        # Objective-aware seeding archive.
        # Maps objective_token -> list of (fitness, genes)
        self._elite_by_objective: Dict[str, List[Tuple[float, List[int]]]] = {}
        self._elite_global: List[Tuple[float, List[int]]] = []
        self._last_eval_result = None

        # Active objective token is "what the pipeline will score next".
        self._active_objective_token: Optional[str] = None

    def evaluate(self, solution: Solution) -> float:
        # Contract requirement: never crash; always return a numeric fitness.
        result = self._pipeline.evaluate(solution)
        solution.fitness = float(result.fitness)

        self._last_eval_result = result
        # Update active objective token (use next if present; otherwise current).
        if result.next_objective_token is not None:
            self._active_objective_token = str(result.next_objective_token)
        elif result.objective_token is not None:
            self._active_objective_token = str(result.objective_token)

        # Archive elites under the objective that was just scored.
        token = result.objective_token
        if isinstance(token, str) and token:
            genes = self._extract_genes(solution)
            fit = float(solution.fitness)
            self._insert_elite(self._elite_by_objective, token, fit, genes)
            self._insert_elite_global(fit, genes)
        return float(result.fitness)

    def get_initial_solution(self) -> Solution:
        genes = self._rng.integers(0, 100, size=self._cfg.dimension).tolist()
        sol = Solution({"genes": genes}, self)
        sol.evaluate()
        return sol

    def get_initial_population(self, population_size: int) -> List[Solution]:
        return [self.get_initial_solution() for _ in range(max(1, int(population_size)))]

    def get_active_objective_token(self) -> Optional[str]:
        """Return the objective token that the pipeline will score next."""

        return self._active_objective_token

    def get_population_seeds(self, *, max_seeds: int = 8) -> List[Solution]:
        """Return seed solutions for the current objective.

        Policy (minimal, deterministic):
        - prefer seeds from a related parent objective when method objective is inherited
        - then seeds from the same objective if previously visited
        - then global elites
        """

        token = self.get_active_objective_token()
        if not token:
            return []

        seeds: List[List[int]] = []

        # If token is a method objective_id, try to seed from the "parent receiver" variant.
        if "|" in token:
            parts = token.split("|", 3)
            if len(parts) == 4:
                receiver, declaring, name, desc = parts
                if receiver and declaring and receiver != declaring:
                    parent_token = f"{declaring}|{declaring}|{name}|{desc}"
                    seeds.extend([g for _f, g in (self._elite_by_objective.get(parent_token) or [])])

        seeds.extend([g for _f, g in (self._elite_by_objective.get(token) or [])])
        seeds.extend([g for _f, g in self._elite_global])

        # Deduplicate by gene tuple.
        uniq: List[List[int]] = []
        seen = set()
        for g in seeds:
            t = tuple(int(x) for x in g)
            if t in seen:
                continue
            seen.add(t)
            uniq.append([int(x) for x in g])
            if len(uniq) >= max(1, int(max_seeds)):
                break

        out: List[Solution] = []
        for g in uniq:
            out.append(Solution({"genes": self._normalize_genes(g)}, self))
        return out

    def _extract_genes(self, solution: Solution) -> List[int]:
        rep = getattr(solution, "representation", None)
        if isinstance(rep, dict) and "genes" in rep:
            genes = list(rep.get("genes") or [])
        elif isinstance(rep, list):
            genes = list(rep)
        else:
            genes = []
        return self._normalize_genes(genes)

    def _normalize_genes(self, genes: List[int]) -> List[int]:
        d = max(1, int(self._cfg.dimension))
        g = [int(x) for x in (genes or [])][:d]
        if len(g) < d:
            g = g + [0] * (d - len(g))
        return g

    def _insert_elite(self, store: Dict[str, List[Tuple[float, List[int]]]], token: str, fitness: float, genes: List[int]) -> None:
        cur = list(store.get(token) or [])
        cur.append((float(fitness), list(genes)))
        cur = sorted(cur, key=lambda x: x[0])
        store[token] = cur[:_ELITE_PER_OBJECTIVE]

    def _insert_elite_global(self, fitness: float, genes: List[int]) -> None:
        cur = list(self._elite_global)
        cur.append((float(fitness), list(genes)))
        cur = sorted(cur, key=lambda x: x[0])
        self._elite_global = cur[:_ELITE_PER_OBJECTIVE]

    def get_problem_info(self) -> Dict[str, Any]:
        return {
            "dimension": int(self._cfg.dimension),
            "problem_type": "mixed",  # placeholder; SBST candidates will be structured objects
            "objective": "minimize_1_minus_branch_coverage",
            "sbst": {
                "build_tool": self._cfg.build_tool,
                "project_root": self._cfg.project_root,
                "targets": list(self._cfg.targets),
                "work_dir": self._cfg.work_dir,
                "timeout_seconds": int(self._cfg.timeout_seconds),
                "gating_enabled": bool(self._cfg.gating_enabled),
                "gating_complete_threshold": float(self._cfg.gating_complete_threshold),
                "plateau_window": int(self._cfg.plateau_window),
                "objective_granularity": str(self._cfg.objective_granularity),
            },
        }

    def get_bounds(self) -> Dict[str, float]:
        return dict(self._bounds)
