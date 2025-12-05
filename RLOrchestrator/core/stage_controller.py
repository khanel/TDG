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
    evals_run: int
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
            # Skip termination phase (solver is None)
            if binding.solver is None:
                continue
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

        if action == 1:  # ADVANCE to next phase
            switched = self._advance_stage()

        evals_run = 0
        
        # Check if we've entered termination phase (solver is None)
        current_stage = self.ctx.stages[self.ctx.phase_index]
        in_termination = current_stage.solver is None
        
        # Only run search steps if not in termination phase
        if not in_termination:
            solver = current_stage.solver
            # Correctly account for evaluations: generations * population size
            pop_size = getattr(solver, 'population_size', 1)

            for _ in range(self.ctx.search_steps_per_decision):
                solver.step()
                # A single solver step is one generation, consuming pop_size evaluations
                evals_this_generation = pop_size
                evals_run += evals_this_generation
                self.ctx.search_step_count += evals_this_generation

                if self.ctx.max_search_steps is not None and self.ctx.search_step_count >= self.ctx.max_search_steps:
                    terminated = True
                    break

        self.ctx.decision_count += 1
        
        # Episode terminates when:
        # 1. We enter termination phase (natural end of pipeline)
        # 2. Max search steps exceeded
        terminated = in_termination or terminated or \
                     (self.ctx.max_search_steps is not None and self.ctx.search_step_count >= self.ctx.max_search_steps)
        
        # Truncated if we run out of decision budget without terminating
        truncated = (not terminated) and (self.ctx.decision_count >= self.ctx.max_decision_steps)
        self._update_best()

        return StageStepResult(
            terminated=terminated,
            truncated=truncated,
            switched=switched,
            evals_run=evals_run,
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
        """Advance to the next phase. Returns True if successfully advanced."""
        if self.ctx.phase_index >= len(self.ctx.stages) - 1:
            return False

        # Export population from current solver (if it exists)
        current_solver = self.ctx.current_solver()
        seeds = current_solver.export_population() if current_solver is not None else []
        
        self.ctx.phase_index += 1
        
        # Ingest population into next solver (if it exists - termination phase has no solver)
        next_solver = self.ctx.current_solver()
        if next_solver is not None:
            if hasattr(next_solver, "ingest_population"):
                next_solver.ingest_population(seeds)
            else:
                # Best-effort fallback for solvers that haven't implemented ingest.
                next_solver.population = seeds
                if hasattr(next_solver, "_update_best_solution"):
                    next_solver._update_best_solution()
        return True

    def _update_best(self) -> None:
        """Update best solution by checking all stages with solvers."""
        for stage in self.ctx.stages:
            # Skip termination phase (no solver)
            if stage.solver is None:
                continue
            stage_best = stage.solver.get_best()
            if stage_best:
                if self.ctx.best_solution is None or stage_best < self.ctx.best_solution:
                    self.ctx.best_solution = stage_best.copy(preserve_id=True)
