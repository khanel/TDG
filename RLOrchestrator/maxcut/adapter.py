"""
Max-Cut problem adapter for RLOrchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import importlib.util
from importlib.util import module_from_spec
from pathlib import Path
import sys
import numpy as np

from Core.problem import ProblemInterface, Solution


def _load_maxcut_module():
    path = Path(__file__).resolve().parents[2] / "MaxCut" / "maxcut.py"
    if not path.exists():
        raise FileNotFoundError(f"MaxCut definition not found at {path}")
    module_name = "RLOrchestrator._maxcut_module"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_maxcut_module = _load_maxcut_module()
MaxCutProblem = _maxcut_module.MaxCutProblem
MaxCutSpec = _maxcut_module.MaxCutSpec
generate_random_maxcut = _maxcut_module.generate_random_maxcut


class MaxCutAdapter(ProblemInterface):
    """Adapter that wraps :class:`MaxCutProblem` and can randomize per episode."""

    def __init__(
        self,
        maxcut_problem: Optional[MaxCutProblem] = None,
        *,
        weight_matrix: Optional[Iterable[Iterable[float]]] = None,
        n_nodes: int = 32,
        edge_probability: float = 0.5,
        seed: Optional[int] = 42,
        ensure_connected: bool = False,
    ):
        self._ensure_connected = bool(ensure_connected)
        self._spec = MaxCutSpec(n_nodes=int(n_nodes), edge_probability=float(edge_probability), seed=seed)
        self._rng = np.random.default_rng(seed)
        self._randomizable = maxcut_problem is None and weight_matrix is None

        if maxcut_problem is not None:
            self.maxcut_problem = maxcut_problem
        elif weight_matrix is not None:
            self.maxcut_problem = MaxCutProblem(weight_matrix, seed=seed, ensure_connected=self._ensure_connected)
        else:
            self.maxcut_problem, _ = generate_random_maxcut(self._spec)
            if self._ensure_connected:
                self.maxcut_problem.ensure_connected = True

        self._bounds = {
            "lower_bound": -float(self.maxcut_problem._total_weight),
            "upper_bound": 0.0,
        }

    def evaluate(self, solution: Solution) -> float:
        return self.maxcut_problem.evaluate(solution)

    def get_initial_solution(self) -> Solution:
        return self.maxcut_problem.get_initial_solution()

    def get_initial_population(self, population_size: int) -> List[Solution]:
        return self.maxcut_problem.get_initial_population(population_size)

    def get_problem_info(self) -> Dict[str, Any]:
        info = self.maxcut_problem.get_problem_info()
        info["problem_type"] = "binary"
        return info

    def get_bounds(self) -> Dict[str, float]:
        return dict(self._bounds)

    def regenerate_instance(self) -> bool:
        if not self._randomizable:
            return False
        seed = int(self._rng.integers(0, 2**31))
        spec = MaxCutSpec(
            n_nodes=self._spec.n_nodes,
            edge_probability=self._spec.edge_probability,
            seed=seed,
        )
        self.maxcut_problem, _ = generate_random_maxcut(spec)
        if self._ensure_connected:
            self.maxcut_problem.ensure_connected = True
        self._bounds = {
            "lower_bound": -float(self.maxcut_problem._total_weight),
            "upper_bound": 0.0,
        }
        return True
