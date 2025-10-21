"""
Problem-agnostic RL environment for phased search.
Uses orchestrator and registries for flexibility.
"""

import gymnasium as gym
import numpy as np
from ..core.orchestrator import Orchestrator
from ..core.observation import ObservationComputer
from ..core.reward import RewardComputer


class RLEnvironment(gym.Env):
    """Gym environment for RL-controlled orchestration."""

    def __init__(self, orchestrator: Orchestrator, max_steps: int = 100, reward_clip: float = 1.0):
        super().__init__()
        self.orchestrator = orchestrator
        self.max_steps = max_steps
        self.step_count = 0
        self.action_space = gym.spaces.Discrete(2)  # 0=continue, 1=switch/terminate
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
        self.step_count = 0
        if hasattr(self, "obs_comp") and hasattr(self.obs_comp, "reset"):
            self.obs_comp.reset()
        return self._observe(), {}

    def step(self, action: int):
        phase = self.orchestrator.get_phase()
        prev_best = self.orchestrator.get_best_solution()
        prev_fit = prev_best.fitness if prev_best else float("inf")

        terminated = False
        if action == 1:
            if phase == "exploration":
                self.orchestrator.switch_to_exploitation()
            else:
                terminated = True
        else:
            self.orchestrator.step(1)
            terminated = False

        self.step_count += 1
        done = terminated or self.step_count >= self.max_steps

        curr_best = self.orchestrator.get_best_solution()
        curr_fit = curr_best.fitness if curr_best else float("inf")
        improvement = prev_fit - curr_fit if prev_fit != float("inf") else 0.0

        obs = self._observe()
        reward = self.reward_comp.compute(action, phase, improvement, terminated)
        return obs, reward, done, False, {}

    def _observe(self):
        solver = self.orchestrator.get_current_solver()
        phase = self.orchestrator.get_phase()
        step_ratio = self.step_count / self.max_steps
        return self.obs_comp.compute(solver, phase, step_ratio)
