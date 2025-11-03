"""
Problem-agnostic RL environment for phased search.
Uses orchestrator and registries for flexibility.
"""

import logging
import math
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from ..core.orchestrator import Orchestrator
from ..core.observation import ObservationComputer
from ..core.reward import RewardComputer
from ..core.utils import IntRangeSpec


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
        logger: Optional[logging.Logger] = None,
        log_type: Optional[str] = None,
        problem_name: Optional[str] = None,
        log_dir: str = 'logs',
        session_id: Optional[int] = None,
        emit_init_summary: bool = True,
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self._rng = np.random.default_rng()
        self._max_decision_spec = self._normalize_range(max_decision_steps)
        self._search_step_spec = self._normalize_range(search_steps_per_decision)
        self.max_decision_steps = self._sample_from_spec(self._max_decision_spec)
        self.search_steps_per_decision = self._sample_from_spec(self._search_step_spec)
        self.max_search_steps = int(max_search_steps) if max_search_steps is not None else None
        self.decision_count = 0
        self.search_step_count = 0
        self._last_observation: Optional[np.ndarray] = None
        self.action_space = gym.spaces.Discrete(2)  # 0=continue, 1=switch phase
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        # Ensure a process-local logger exists (robust for SubprocVecEnv)
        try:
            from multiprocessing import current_process
            in_main = current_process().name == 'MainProcess'
        except Exception:
            in_main = True
        if logger is not None and in_main:
            self.logger = logger
        else:
            # Initialize a fresh logger in this process when needed
            from ..core.utils import setup_logging as _setup_logging
            problem_label = str(problem_name) if problem_name is not None else type(orchestrator.problem).__name__.lower()
            type_label = str(log_type) if log_type is not None else 'train'
            self.logger = _setup_logging(type_label, problem_label, log_dir=log_dir, session_id=session_id)
        self._emit_init_summary = bool(emit_init_summary)

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
        self.obs_comp = ObservationComputer(meta, logger=self.logger)
        clip = abs(float(reward_clip))
        self.reward_comp = RewardComputer(meta, clip_range=(-clip, clip), logger=self.logger)

        if self._emit_init_summary:
            try:
                obs_shape = tuple(int(x) for x in (self.observation_space.shape or ()))
                # One-line concise init summary
                self.logger.info(
                    f"Env init: max_decisions={max_decision_steps}, steps_per_decision={search_steps_per_decision}, "
                    f"max_search_steps={max_search_steps}, reward_clip={reward_clip}"
                )
                # Observation space + features (one line)
                obs_names = [
                    "budget_remaining",
                    "normalized_best_fitness",
                    "improvement_velocity",
                    "stagnation_nonparametric",
                    "population_concentration",
                    "landscape_funnel_proxy",
                    "landscape_deceptiveness_proxy",
                    "active_phase",
                ]
                self.logger.info(
                    f"Obs space: shape={obs_shape}, dtype={self.observation_space.dtype}, features={obs_names}"
                )
                # Reward formula (one line)
                try:
                    clip_min = getattr(self.reward_comp, "_clip_min", None)
                    clip_max = getattr(self.reward_comp, "_clip_max", None)
                    eff = getattr(self.reward_comp, "efficiency_penalty", None)
                    self.logger.info(
                        f"Reward: R=(1-B)*progress + B*exploration - efficiency_penalty + decision_bonus; "
                        f"clip=[{clip_min},{clip_max}], efficiency_penalty={eff}"
                    )
                except Exception:
                    pass
            except Exception:
                pass

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()
        self.max_decision_steps = self._sample_from_spec(self._max_decision_spec)
        self.search_steps_per_decision = self._sample_from_spec(self._search_step_spec)
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
        improvement = 0.0
        steps_run = 0
        switched = False

        if action == 1:
            if phase == "exploration":
                self.orchestrator.switch_to_exploitation()
                switched = True
            else:
                terminated = True

        if not terminated:
            for _ in range(self.search_steps_per_decision):
                self.orchestrator.step(1)
                steps_run += 1
                self.search_step_count += 1
                if self.max_search_steps is not None and self.search_step_count >= self.max_search_steps:
                    truncated = True
                    break
        current_phase = self.orchestrator.get_phase()
        curr_best = self.orchestrator.get_best_solution()
        curr_fit = curr_best.fitness if curr_best else float("inf")
        if math.isfinite(prev_fit) and math.isfinite(curr_fit):
            improvement = prev_fit - curr_fit

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
            steps_run=steps_run,
            switched=switched,
            phase_after=current_phase,
        )
        self.logger.debug(f"Step {self.decision_count}: action={action}, reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        return obs, reward, terminated, truncated, {}

    def _observe(self):
        solver = self.orchestrator.get_current_solver()
        phase = self.orchestrator.get_phase()
        step_ratio = self.decision_count / self.max_decision_steps
        observation = self.obs_comp.compute(solver, phase, step_ratio)
        self.logger.debug(f"Observation at step {self.decision_count}: {observation}")
        return observation

    def _normalize_range(self, spec: IntRangeSpec | int) -> Tuple[int, int]:
        if isinstance(spec, tuple):
            lo, hi = int(spec[0]), int(spec[1])
        elif isinstance(spec, list) and len(spec) == 2:
            lo, hi = int(spec[0]), int(spec[1])
        else:
            value = int(spec)
            return (max(1, value), max(1, value))
        lo, hi = sorted((lo, hi))
        lo = max(1, lo)
        hi = max(lo, hi)
        return (lo, hi)

    def _sample_from_spec(self, spec: Tuple[int, int]) -> int:
        lo, hi = spec
        if lo == hi:
            return lo
        return int(self._rng.integers(lo, hi + 1))
