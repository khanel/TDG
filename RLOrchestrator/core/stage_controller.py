"""
Domain logic for sequencing exploration/exploitation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from Core.problem import Solution

from .context import (
    OrchestratorContext,
    Phase,
    normalize_range,
    sample_from_range,
)


@dataclass
class StageStepResult:
    terminated: bool
    truncated: bool
    switched: bool
    steps_run: int
    prev_phase: Phase
    phase_after: Phase
    prev_best: Optional[Solution]
    curr_best: Optional[Solution]


class StageController:
    """State machine that coordinates solver progression."""

    def __init__(self, context: OrchestratorContext):
        self.ctx = context
        # TODO: add unit tests covering reset/step semantics to protect against regressions.
        self._max_decision_spec = normalize_range(context.budget.max_decision_steps)
        self._search_step_spec = normalize_range(context.budget.search_steps_per_decision)
        self._max_search_steps = (
            None if context.budget.max_search_steps is None
            else int(context.budget.max_search_steps)
        )

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset episode state and reinitialize solvers."""
        if seed is not None:
            self.ctx.rng = np.random.default_rng(seed)

        self.ctx.reset_state()
        self.ctx.max_decision_steps = sample_from_range(self._max_decision_spec, self.ctx.rng)
        self.ctx.search_steps_per_decision = sample_from_range(self._search_step_spec, self.ctx.rng)
        self.ctx.max_search_steps = self._max_search_steps

        instance_regenerated = False
        if hasattr(self.ctx.problem, "regenerate_instance"):
            try:
                instance_regenerated = bool(self.ctx.problem.regenerate_instance())
            except Exception:
                instance_regenerated = False

        for binding in self.ctx.stages:
            # Always initialize after regenerating the instance to keep solvers in sync
            if hasattr(binding.solver, "initialize"):
                binding.solver.initialize()
            elif instance_regenerated and hasattr(binding.solver, "ingest_population"):
                binding.solver.ingest_population([])

        self._update_best()

    def step(self, action: int) -> StageStepResult:
        prev_phase = self.ctx.current_phase()
        prev_best = self.ctx.best_solution
        terminated = False
        switched = False

        if action == 1:  # ADVANCE
            switched = self._advance_stage()
            if not switched:
                terminated = True

        steps_run = 0
        if not terminated:
            solver = self.ctx.current_solver()
            for _ in range(self.ctx.search_steps_per_decision):
                solver.step()
                steps_run += 1
                self.ctx.search_step_count += 1
                if self.ctx.max_search_steps is not None and self.ctx.search_step_count >= self.ctx.max_search_steps:
                    terminated = True
                    break

        self.ctx.decision_count += 1
        truncated = self.ctx.decision_count >= self.ctx.max_decision_steps
        self._update_best()

        return StageStepResult(
            terminated=terminated,
            truncated=truncated,
            switched=switched,
            steps_run=steps_run,
            prev_phase=prev_phase,
            phase_after=self.ctx.current_phase(),
            prev_best=prev_best,
            curr_best=self.ctx.best_solution,
        )

    def current_solver(self):
        return self.ctx.current_solver()

    def current_phase(self) -> Phase:
        return self.ctx.current_phase()

    def get_best_solution(self) -> Optional[Solution]:
        return self.ctx.best_solution

    def _advance_stage(self) -> bool:
        if self.ctx.phase_index >= len(self.ctx.stages) - 1:
            return False

        seeds = self.ctx.current_solver().export_population()
        self.ctx.phase_index += 1
        next_solver = self.ctx.current_solver()
        if hasattr(next_solver, "ingest_population"):
            next_solver.ingest_population(seeds)
        else:
            # Best-effort fallback for solvers that haven't implemented ingest.
            next_solver.population = seeds
            if hasattr(next_solver, "_update_best_solution"):
                next_solver._update_best_solution()
        return True

    def _update_best(self) -> None:
        exp_best = None
        exp_solver = self.ctx.stages[0].solver if self.ctx.stages else None
        if exp_solver is not None:
            exp_best = exp_solver.get_best()
        exp_fit = exp_best.fitness if exp_best else float("inf")

        exploit_solver = None
        if len(self.ctx.stages) > 1 and self.ctx.phase_index > 0:
            exploit_solver = self.ctx.current_solver()
        exploit_best = exploit_solver.get_best() if exploit_solver else None
        exploit_fit = exploit_best.fitness if exploit_best else float("inf")

        current_best = self.ctx.best_solution
        current_fit = current_best.fitness if current_best else float("inf")

        if exp_fit < current_fit:
            self.ctx.best_solution = exp_best
            current_fit = exp_fit

        if exploit_fit < current_fit:
            self.ctx.best_solution = exploit_best
