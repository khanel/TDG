"""
Integrated, problem-agnostic RL environment for phased search orchestration.
This file combines the core state machine and the Gym environment interface.
"""

import logging
import math
from typing import Optional, List, Literal, Tuple

import gymnasium as gym
import numpy as np

from Core.problem import Solution, ProblemInterface
from Core.search_algorithm import SearchAlgorithm
from .observation import ObservationComputer
from .reward import RewardComputer
from .utils import IntRangeSpec, setup_logging

Phase = Literal["exploration", "exploitation"]


class OrchestratorEnv(gym.Env):
    """
    A uni-directional, two-stage RL environment for solver orchestration.
    It learns a policy to decide the optimal time to switch from an
    exploration solver to an exploitation solver.
    """

    def __init__(
        self,
        problem: ProblemInterface,
        exploration_solver: SearchAlgorithm,
        exploitation_solver: SearchAlgorithm,
        max_decision_steps: IntRangeSpec = 100,
        *,
        search_steps_per_decision: IntRangeSpec = 1,
        max_search_steps: Optional[int] = None,
        reward_clip: float = 1.0,
        log_type: str = 'train',
        log_dir: str = 'logs',
        session_id: Optional[int] = None,
        emit_init_summary: bool = True,
    ):
        super().__init__()
        self.problem = problem
        self.exploration_solver = exploration_solver
        self.exploitation_solver = exploitation_solver
        self.phase: Phase = "exploration"
        self._best_solution: Optional[Solution] = None

        self._rng = np.random.default_rng()
        self._max_decision_spec = self._normalize_range(max_decision_steps)
        self._search_step_spec = self._normalize_range(search_steps_per_decision)
        self.max_decision_steps = self._sample_from_spec(self._max_decision_spec)
        self.search_steps_per_decision = self._sample_from_spec(self._search_step_spec)
        self.max_search_steps = int(max_search_steps) if max_search_steps is not None else None
        
        self.decision_count = 0
        self.search_step_count = 0
        self._last_observation: Optional[np.ndarray] = None

        self.action_space = gym.spaces.Discrete(2)  # 0=STAY, 1=ADVANCE
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        problem_label = type(problem).__name__.lower()
        self.logger = setup_logging(log_type, problem_label, log_dir=log_dir, session_id=session_id)

        meta = self._get_problem_meta()
        self.obs_comp = ObservationComputer(meta, logger=self.logger)
        clip = abs(float(reward_clip))
        self.reward_comp = RewardComputer(meta, clip_range=(-clip, clip), logger=self.logger)

        if emit_init_summary:
            self._log_init_summary(max_decision_steps, search_steps_per_decision, max_search_steps, reward_clip)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.max_decision_steps = self._sample_from_spec(self._max_decision_spec)
        self.search_steps_per_decision = self._sample_from_spec(self._search_step_spec)
        self.decision_count = 0
        self.search_step_count = 0
        
        if hasattr(self.problem, "regenerate_instance") and self.problem.regenerate_instance():
            for solver in (self.exploration_solver, self.exploitation_solver):
                if hasattr(solver, "initialize"):
                    solver.initialize()
        
        self.phase = "exploration"
        self._best_solution = None
        self._update_best()
        
        if hasattr(self.obs_comp, "reset"):
            self.obs_comp.reset()
            
        obs = self._observe()
        self._last_observation = obs.copy()
        return obs, {}

    def step(self, action: int):
        prev_phase = self.phase
        prev_best = self._best_solution
        prev_fit = prev_best.fitness if prev_best else float("inf")
        prev_observation = self._last_observation

        terminated = False
        switched = False

        if action == 1:  # ADVANCE
            if self.phase == "exploration":
                self._switch_to_exploitation()
                switched = True
            else: # Was in exploitation, so advancing terminates
                terminated = True
        
        steps_run = 0
        if not terminated:
            current_solver = self.get_current_solver()
            for _ in range(self.search_steps_per_decision):
                current_solver.step()
                steps_run += 1
                self.search_step_count += 1
                if self.max_search_steps is not None and self.search_step_count >= self.max_search_steps:
                    terminated = True  # Budget exhausted
                    break
        
        self._update_best()
        curr_best = self._best_solution
        curr_fit = curr_best.fitness if curr_best else float("inf")
        improvement = prev_fit - curr_fit if math.isfinite(prev_fit) and math.isfinite(curr_fit) else 0.0

        self.decision_count += 1
        truncated = self.decision_count >= self.max_decision_steps

        obs = self._observe()
        self._last_observation = obs.copy()
        
        reward = self.reward_comp.compute(
            action=action,
            phase=prev_phase,
            improvement=improvement,
            terminated=terminated,
            observation=obs,
            prev_observation=prev_observation,
            steps_run=steps_run,
            switched=switched,
            phase_after=self.phase,
        )
        
        self.logger.debug(f"Step {self.decision_count}: action={action}, reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        
        return obs, reward, terminated, truncated, {}

    def get_phase(self) -> Phase:
        return self.phase

    def get_current_solver(self) -> SearchAlgorithm:
        return self.exploration_solver if self.phase == "exploration" else self.exploitation_solver

    def get_best_solution(self) -> Optional[Solution]:
        return self._best_solution

    def _observe(self) -> np.ndarray:
        solver = self.get_current_solver()
        step_ratio = self.decision_count / self.max_decision_steps if self.max_decision_steps > 0 else 0
        observation = self.obs_comp.compute(solver, self.phase, step_ratio)
        self.logger.debug(f"Observation at step {self.decision_count}: {observation}")
        return observation

    def _switch_to_exploitation(self, seeds: Optional[List[Solution]] = None) -> None:
        if self.phase == "exploitation":
            return
        
        if seeds is None:
            pop = self.exploration_solver.get_population()
            seeds = [s for s in pop if s is not None]
            
        if hasattr(self.exploitation_solver, "ingest_seeds"):
            self.exploitation_solver.ingest_seeds(seeds)
        else:
            self.exploitation_solver.population = [s.copy() for s in seeds]
            if hasattr(self.exploitation_solver, "_update_best_solution"):
                self.exploitation_solver._update_best_solution()
                
        self.phase = "exploitation"
        self._update_best()

    def _update_best(self) -> None:
        exp_best = self.exploration_solver.get_best()
        exp_fit = exp_best.fitness if exp_best else float("inf")
        
        # Only consider exploitation solver's fitness if it has been used
        if self.phase == "exploitation":
            exploit_best = self.exploitation_solver.get_best()
            exploit_fit = exploit_best.fitness if exploit_best else float("inf")
        else:
            exploit_best = None
            exploit_fit = float("inf")

        current_best = self._best_solution
        current_fit = current_best.fitness if current_best else float("inf")

        if exp_fit < current_fit:
            self._best_solution = exp_best
            current_fit = exp_fit
        
        if exploit_fit < current_fit:
            self._best_solution = exploit_best

    def _get_problem_meta(self) -> dict:
        meta = {}
        if hasattr(self.problem, "get_bounds"):
            try: meta.update(self.problem.get_bounds() or {})
            except Exception: pass
        if hasattr(self.problem, "get_problem_info"):
            try:
                info = self.problem.get_problem_info() or {}
                if isinstance(info, dict): meta.update(info)
            except Exception: pass
        return meta

    def _normalize_range(self, spec: IntRangeSpec | int) -> Tuple[int, int]:
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            lo, hi = int(spec[0]), int(spec[1])
        else:
            value = int(spec)
            return (max(1, value), max(1, value))
        lo, hi = sorted((lo, hi))
        return (max(1, lo), max(lo, hi))

    def _sample_from_spec(self, spec: Tuple[int, int]) -> int:
        lo, hi = spec
        return int(self._rng.integers(lo, hi + 1)) if lo < hi else lo

    def _log_init_summary(self, max_dec, search_steps, max_search, reward_clip):
        obs_shape = tuple(int(x) for x in (self.observation_space.shape or ()))
        self.logger.info(
            f"Env init: max_decisions={max_dec}, steps_per_decision={search_steps}, "
            f"max_search_steps={max_search}, reward_clip={reward_clip}"
        )
        obs_names = getattr(self.obs_comp, 'feature_names', ["-"] * obs_shape[0])
        self.logger.info(
            f"Obs space: shape={obs_shape}, dtype={self.observation_space.dtype}, features={obs_names}"
        )
        try:
            clip_min, clip_max = self.reward_comp._clip_min, self.reward_comp._clip_max
            eff = getattr(self.reward_comp, "efficiency_penalty", "N/A")
            self.logger.info(
                "Reward: R = w_q(B)*prog + w_e(B)*diversity - efficiency - penalties; "
                f"clip=[{clip_min},{clip_max}], efficiency_penalty={eff}"
            )
        except Exception:
            pass