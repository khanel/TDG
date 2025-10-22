"""
Problem-agnostic RL environment for phased search.
Uses orchestrator and registries for flexibility.
"""

import gymnasium as gym
import numpy as np
from typing import Optional
from ..core.orchestrator import Orchestrator
from ..core.observation import ObservationComputer
from ..core.reward import RewardComputer


class RLEnvironment(gym.Env):
    """Gym environment for RL-controlled orchestration."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        max_decision_steps: int = 100,
        *,
        search_steps_per_decision: int = 1,
        max_search_steps: Optional[int] = None,
        reward_clip: float = 1.0,
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.max_decision_steps = max(1, int(max_decision_steps))
        self.search_steps_per_decision = max(1, int(search_steps_per_decision))
        self.max_search_steps = int(max_search_steps) if max_search_steps is not None else None
        self.decision_count = 0
        self.search_step_count = 0
        self._last_observation: Optional[np.ndarray] = None
        self.action_space = gym.spaces.Discrete(2)  # 0=continue, 1=switch phase
        # Observation vector (7):
        # [phase_is_exploitation, normalized_best_fitness, frontier_improvement_flag,
        #  frontier_success_rate, elite_turnover_entropy, frontier_stagnation_ratio, budget_used_ratio]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        # Build normalization meta from problem, preferring explicit bounds if available
        meta = {}
        problem_obj = orchestrator.problem
        if hasattr(problem_obj, "get_bounds"):
            try:
                meta.update(problem_obj.get_bounds() or {})
            except Exception:
                pass
        if hasattr(problem_obj, "get_problem_info"):
            try:
                info = problem_obj.get_problem_info() or {}
                if isinstance(info, dict):
                    meta.update(info)
            except Exception:
                pass
        self.obs_comp = ObservationComputer(meta)
        clip = abs(float(reward_clip))
        self.reward_comp = RewardComputer(meta, clip_range=(-clip, clip))

    def reset(self, *, seed=None, options=None):
        self.decision_count = 0
        self.search_step_count = 0
        if hasattr(self.orchestrator.problem, "regenerate_instance"):
            try:
                regenerated = bool(self.orchestrator.problem.regenerate_instance())
            except Exception:
                regenerated = False
            if regenerated:
                for solver in (self.orchestrator.exploration_solver, self.orchestrator.exploitation_solver):
                    if hasattr(solver, "initialize"):
                        solver.initialize()
                self.orchestrator.phase = "exploration"
                self.orchestrator._best_solution = None
                self.orchestrator._update_best()
        if hasattr(self, "obs_comp") and hasattr(self.obs_comp, "reset"):
            self.obs_comp.reset()
        obs = self._observe()
        self._last_observation = obs.copy()
        return obs, {}

    def step(self, action: int):
        phase = self.orchestrator.get_phase()
        prev_best = self.orchestrator.get_best_solution()
        prev_fit = prev_best.fitness if prev_best else float("inf")
        prev_observation = self._last_observation

        terminated = False
        truncated = False
        if action == 1:
            if phase == "exploration":
                self.orchestrator.switch_to_exploitation()
            else:
                terminated = True

        improvement = 0.0
        curr_best = prev_best
        curr_fit = prev_fit
        if not terminated:
            steps_run = 0
            for _ in range(self.search_steps_per_decision):
                self.orchestrator.step(1)
                steps_run += 1
                self.search_step_count += 1
                if self.max_search_steps is not None and self.search_step_count >= self.max_search_steps:
                    truncated = True
                    break
            curr_best = self.orchestrator.get_best_solution()
            curr_fit = curr_best.fitness if curr_best else float("inf")
            improvement = prev_fit - curr_fit if (prev_fit != float("inf") and curr_fit != float("inf")) else 0.0

        self.decision_count += 1
        if not terminated and self.decision_count >= self.max_decision_steps:
            truncated = True

        obs = self._observe()
        self._last_observation = obs.copy()
        reward = self.reward_comp.compute(
            action=action,
            phase=phase,
            improvement=improvement,
            terminated=terminated,
            observation=obs,
            prev_observation=prev_observation,
        )
        return obs, reward, terminated, truncated, {}

    def _observe(self):
        solver = self.orchestrator.get_current_solver()
        phase = self.orchestrator.get_phase()
        step_ratio = self.decision_count / self.max_decision_steps
        return self.obs_comp.compute(solver, phase, step_ratio)
