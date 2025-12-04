"""
Integrated, problem-agnostic RL environment for phased search orchestration.
This version separates the Gym wrapper (policy-facing) from the domain controller
that manages solver progression.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import gymnasium as gym
import numpy as np

from Core.problem import Solution, ProblemInterface
from Core.search_algorithm import SearchAlgorithm
from .context import BudgetSpec, OrchestratorContext, Phase, StageBinding
from .observation import ObservationComputer, ObservationState
from .stage_controller import StageController
from .utils import IntRangeSpec, setup_logging
from .reward import RewardWrapper


class OrchestratorEnv(gym.Env):
    """
    A uni-directional, two-stage RL environment for solver orchestration.
    It learns a policy to decide the optimal time to switch from an
    exploration solver to an exploitation solver.
    """

    metadata = {"render.modes": []}

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
        self._last_observation: Optional[np.ndarray] = None

        self.action_space = gym.spaces.Discrete(2)  # 0=STAY, 1=ADVANCE

        problem_label = type(problem).__name__.lower()
        self.logger = setup_logging(log_type, problem_label, log_dir=log_dir, session_id=session_id)

        meta = self._get_problem_meta()
        self.obs_comp = ObservationComputer(meta, logger=self.logger)

        obs_space_size = len(self.obs_comp.feature_names)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_space_size,), dtype=np.float32)

        budget = BudgetSpec(
            max_decision_steps=max_decision_steps,
            search_steps_per_decision=search_steps_per_decision,
            max_search_steps=max_search_steps,
        )
        context = OrchestratorContext(
            problem=problem,
            stages=[
                StageBinding(name="exploration", solver=exploration_solver),
                StageBinding(name="exploitation", solver=exploitation_solver),
                StageBinding(name="termination", solver=None),  # Terminal phase - no solver
            ],
            budget=budget,
        )
        self._context = context
        self._controller = StageController(context)
        self.reward_computer = RewardWrapper()

        if emit_init_summary:
            self._log_init_summary(max_decision_steps, search_steps_per_decision, max_search_steps, reward_clip)

    # Gym API ---------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self._controller.reset(seed=seed)
        if hasattr(self.obs_comp, "reset"):
            self.obs_comp.reset()

        obs = self._observe()
        self._last_observation = obs.copy()
        return obs, {}

    def step(self, action: int):
        prev_observation = self._last_observation
        result = self._controller.step(action)

        obs = self._observe()
        self._last_observation = obs.copy()

        reward_signal = self.reward_computer.compute(prev_observation, action, {})
        reward = float(np.clip(reward_signal.value, -self.reward_clip, self.reward_clip))

        self.logger.debug(
            "Step %s: action=%s reward=%.4f terminated=%s truncated=%s phase=%s",
            self._context.decision_count,
            action,
            reward,
            result.terminated,
            result.truncated,
            result.phase_after,
        )

        return obs, reward, result.terminated, result.truncated, {}

    # Convenience accessors ------------------------------------------------

    def get_phase(self) -> Phase:
        return self._controller.current_phase()

    def get_current_solver(self) -> SearchAlgorithm:
        return self._controller.current_solver()

    def get_best_solution(self) -> Optional[Solution]:
        return self._controller.get_best_solution()

    # Internals ------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        solver = self.get_current_solver()
        phase = self.get_phase()
        max_decisions = max(1, self._context.max_decision_steps or 1)
        step_ratio = self._context.decision_count / max_decisions if max_decisions > 0 else 0.0
        state = ObservationState(
            solver=solver,
            phase=phase,
            step_ratio=step_ratio,
            best_solution=solver.get_best(),
            population=solver.get_population(),
        )
        observation = self.obs_comp.compute(state)
        self.logger.debug(f"Observation at decision {self._context.decision_count}: {observation}")
        return observation

    def _get_problem_meta(self) -> dict:
        meta: dict = {}
        try:
            bounds = self.problem.get_bounds()
            if isinstance(bounds, dict):
                meta.update(bounds)
        except Exception:
            pass
        try:
            info = self.problem.get_problem_info()
            if isinstance(info, dict):
                meta.update(info)
        except Exception:
            pass
        return meta

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
