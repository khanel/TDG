"""
Temporary PPO training harness using the ElasticRewardComputer with elastic
stagnation pressure to balance effectiveness and efficiency.

Default training problem is switched to NKL for faster iterations; evaluation
can still target TSP via the separate evaluator script.
"""

import argparse
from dataclasses import dataclass
from typing import Optional
import sys
from pathlib import Path
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Ensure project root is on PYTHONPATH for relative imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if torch.cuda.is_available():
    # Allow TF32 for faster matmul on recent GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from RLOrchestrator.core.orchestrator import OrchestratorEnv
from RLOrchestrator.problems.registry import get_problem_definition, SolverFactory
from RLOrchestrator.core.reward import ElasticRewardComputer, RewardConfig


class ElasticRewardOrchestratorEnv(OrchestratorEnv):
    """OrchestratorEnv that uses the elastic stagnation pressure reward."""

    def __init__(
        self,
        *,
        reward_config: RewardConfig,
        max_decision_steps: int = 50,
        search_steps_per_decision: int = 1,
        max_search_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            max_decision_steps=max_decision_steps,
            search_steps_per_decision=search_steps_per_decision,
            max_search_steps=max_search_steps,
            reward_clip=1e6,  # Clipping is less critical with a well-shaped reward
            **kwargs,
        )
        self.reward_comp = ElasticRewardComputer(reward_config)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.reward_comp.reset(initial_norm_best=float(obs[1]))
        return obs, info

    def step(self, action: int):
        # Manually drive the controller to bypass the base class's flawed reward calculation
        result = self._controller.step(action)

        # The controller now correctly reports the evaluations used in the step
        evals_used_this_step = result.evals_run

        obs = self._observe()
        self._last_observation = obs.copy()

        total_budget = self._compute_total_budget()
        total_decision_steps = self._context.max_decision_steps or 1

        # Compute reward using the ElasticRewardComputer
        reward = self.reward_comp.compute(
            observation=obs,
            evals_used_this_step=evals_used_this_step,
            total_budget=total_budget,
            total_decision_steps=total_decision_steps,
            terminated=(result.terminated or result.truncated)
        )

        # Pass debug info
        info = {
            "evals_used": evals_used_this_step,
            "total_evals": self._context.search_step_count,
            "reward": reward,
        }

        return obs, reward, result.terminated, result.truncated, info

    def _compute_total_budget(self) -> int:
        """Derive the search-step budget for cost penalties."""
        if self._context.max_search_steps is not None:
            return max(1, int(self._context.max_search_steps))
        return max(
            1,
            int(self._context.max_decision_steps)
            * max(1, int(self._context.search_steps_per_decision)),
        )


def _parse_int_or_range(value: str | int | float, *, minimum: int = 1) -> int | tuple[int, int]:
    """Parse scalar or simple range strings like '50-150' into int or (lo, hi)."""
    if isinstance(value, (int, float)):
        return max(minimum, int(value))
    text = str(value).strip()
    for sep in ("-", ":", ","):
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) == 2:
                lo, hi = sorted((int(float(parts[0])), int(float(parts[1]))))
                return (max(minimum, lo), max(minimum, hi))
    return max(minimum, int(float(text)))


def _sample_from_spec(spec: int | float | tuple[int, int], *, minimum: int = 1) -> int:
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        return int(np.random.randint(max(minimum, lo), max(minimum + 1, hi + 1)))
    return int(max(minimum, int(spec)))


def _sample_solver_pair(problem_name: str) -> tuple[SolverFactory, SolverFactory]:
    definition = get_problem_definition(problem_name)
    if definition is None:
        raise KeyError(f"Problem '{problem_name}' is not registered.")

    def as_list(spec):
        if isinstance(spec, (list, tuple)):
            return [s for s in spec if isinstance(s, SolverFactory)]
        return [spec] if isinstance(spec, SolverFactory) else []

    explorers = as_list(definition.solvers.get("exploration", []))
    exploiters = as_list(definition.solvers.get("exploitation", []))
    if not explorers or not exploiters:
        raise ValueError("Problem definition must include exploration and exploitation solver factories.")
    return np.random.choice(explorers), np.random.choice(exploiters)


def build_env(
    problem_name: str = "nkl",
    reward_config: RewardConfig = RewardConfig(),
    max_decision_steps: Optional[int] | tuple[int, int] = 50,
    search_steps_per_decision: int = 1,
    max_search_steps: Optional[int] = None,
    adapter_kwargs: Optional[dict] = None,
    solver_kwargs: Optional[dict] = None,
    solver_pair: Optional[tuple[SolverFactory, SolverFactory]] = None,
    log_type: str = "elastic_reward",
    explorer_pop_scale: float = 1.0,
    exploiter_pop_scale: float = 1.0,
):
    """
    Instantiate a problem bundle and wrap it in the elastic reward environment.
    """
    if solver_pair is None:
        solver_pair = _sample_solver_pair(problem_name)
    exp_factory, exploit_factory = solver_pair

    definition = get_problem_definition(problem_name)
    if definition is None:
        raise KeyError(f"Problem '{problem_name}' is not registered.")

    adapter_params = dict(definition.default_adapter_kwargs)
    if adapter_kwargs:
        adapter_params.update(adapter_kwargs)
    adapter = definition.adapter_cls(**adapter_params)

    solver_kwargs = solver_kwargs or {}
    explorer = exp_factory.build(adapter, solver_kwargs.get("exploration"))
    exploiter = exploit_factory.build(adapter, solver_kwargs.get("exploitation"))

    if hasattr(explorer, "population_size"):
        explorer.population_size = max(1, int(explorer.population_size * float(explorer_pop_scale)))
    if hasattr(exploiter, "population_size"):
        exploiter.population_size = max(1, int(exploiter.population_size * float(exploiter_pop_scale)))

    return ElasticRewardOrchestratorEnv(
        problem=adapter,
        exploration_solver=explorer,
        exploitation_solver=exploiter,
        reward_config=reward_config,
        max_decision_steps=max_decision_steps,
        search_steps_per_decision=search_steps_per_decision,
        max_search_steps=max_search_steps,
        log_type=log_type,
        emit_init_summary=False,
    )


def _build_solver_sweep_vec_env(
    problem_name: str,
    reward_config: RewardConfig,
    max_decision_steps: int | tuple[int, int],
    search_steps_per_decision: int | tuple[int, int],
    max_search_steps: Optional[int],
    adapter_kwargs: Optional[dict],
    solver_kwargs: Optional[dict],
    num_envs: int,
    log_type: str,
    explorer_pop_scale: float,
    exploiter_pop_scale: float,
):
    env_fns = []

    def make_env():
        return build_env(
            problem_name=problem_name,
            reward_config=reward_config,
            max_decision_steps=max_decision_steps,
            search_steps_per_decision=search_steps_per_decision,
            max_search_steps=max_search_steps,
            adapter_kwargs=adapter_kwargs,
            solver_kwargs=solver_kwargs,
            solver_pair=_sample_solver_pair(problem_name),
            log_type=log_type,
            explorer_pop_scale=explorer_pop_scale,
            exploiter_pop_scale=exploiter_pop_scale,
        )

    for _ in range(max(1, num_envs)):
        env_fns.append(make_env)
    if num_envs > 1:
        return SubprocVecEnv(env_fns, start_method="fork")
    return DummyVecEnv(env_fns)


def run_probe(
    *,
    problem_name: str = "nkl",
    total_timesteps: int = 5_000,
    model_path: str = "temp/ppo_elastic_reward.zip",
    resume_model_path: Optional[str] = None,
    max_decision_steps: int = 50,
    search_steps_per_decision: int | tuple[int, int] = 1,
    max_search_steps: Optional[int] = None,
    reward_config: RewardConfig = RewardConfig(), # Uses new defaults
    adapter_kwargs: Optional[dict] = None,
    solver_kwargs: Optional[dict] = None,
    device: str = "cpu",
    disable_omp: bool = True,
    num_envs: int = 4,
    log_type: str = "elastic_reward",
    ppo_n_steps: int = 512,
    ppo_batch_size: int = 4096,
    ppo_epochs: int = 2,
    ppo_lr: float = 3e-4,
    ppo_ent_coef: float = 0.01,
    explorer_pop_scale: float = 1.0,
    exploiter_pop_scale: float = 1.0,
    progress_bar: bool = False,
):
    if disable_omp:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)

    env = _build_solver_sweep_vec_env(
        problem_name=problem_name,
        reward_config=reward_config,
        max_decision_steps=max_decision_steps,
        search_steps_per_decision=search_steps_per_decision,
        max_search_steps=max_search_steps,
        adapter_kwargs=adapter_kwargs,
        solver_kwargs=solver_kwargs,
        num_envs=num_envs,
        log_type=log_type,
        explorer_pop_scale=explorer_pop_scale,
        exploiter_pop_scale=exploiter_pop_scale,
    )

    resume_path = Path(resume_model_path) if resume_model_path else None
    if resume_path and resume_path.exists():
        model = PPO.load(resume_path, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            n_epochs=ppo_epochs,
            learning_rate=ppo_lr,
            ent_coef=ppo_ent_coef,
            policy_kwargs={"net_arch": [64, 64]},
        )
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--model-path", type=str, default="temp/ppo_elastic_reward.zip")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-decision-steps", type=str, default="40-160")
    parser.add_argument("--search-steps-per-decision", type=str, default="100") # Increased for meaningful steps
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--problem-name", type=str, default="nkl", help="Problem to train on (default: nkl for speed)")
    parser.add_argument("--tsp-num-cities", type=str, default="30-120")
    parser.add_argument("--tsp-grid-size", type=float, default=120.0)
    parser.add_argument("--nkl-n-items", type=str, default="80-160", help="Range or int for NKL n_items")
    parser.add_argument("--nkl-k-interactions", type=str, default="2-10", help="Range or int for NKL k_interactions")
    parser.add_argument("--resume-model", type=str, default=None, help="Path to an existing PPO model to continue training")
    parser.add_argument("--num-envs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--ppo-n-steps", type=int, default=256)
    parser.add_argument("--ppo-batch-size", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01)
    parser.add_argument("--log-type", type=str, default="elastic_reward")
    parser.add_argument("--explorer-pop-scale", type=float, default=1.0)
    parser.add_argument("--exploiter-pop-scale", type=float, default=1.0)
    parser.add_argument("--progress-bar", action="store_true", help="Enable SB3 progress bar (slower).")
    args = parser.parse_args()

    # Build adapter kwargs with problem-specific randomization
    adapter_kwargs = None
    problem_lower = args.problem_name.lower()
    if problem_lower == "tsp":
        adapter_kwargs = {
            "num_cities": _parse_int_or_range(args.tsp_num_cities, minimum=3),
            "grid_size": args.tsp_grid_size,
        }
    elif problem_lower == "nkl":
        n_items_spec = _parse_int_or_range(args.nkl_n_items, minimum=2)
        k_interactions_spec = _parse_int_or_range(args.nkl_k_interactions, minimum=0)
        n_items = _sample_from_spec(n_items_spec, minimum=2)
        k_interactions = min(_sample_from_spec(k_interactions_spec, minimum=0), max(1, n_items - 1))
        adapter_kwargs = {
            "n_items": n_items,
            "k_interactions": k_interactions,
        }
    search_steps = _parse_int_or_range(args.search_steps_per_decision, minimum=1)
    max_decisions = _parse_int_or_range(args.max_decision_steps, minimum=10)

    # Use the elastic reward defaults (tune in one place: RLOrchestrator/core/reward.py)
    reward_cfg = RewardConfig()

    run_probe(
        problem_name=args.problem_name,
        total_timesteps=args.total_timesteps,
        model_path=args.model_path,
        resume_model_path=args.resume_model,
        max_decision_steps=max_decisions,
        search_steps_per_decision=search_steps,
        max_search_steps=args.max_search_steps,
        adapter_kwargs=adapter_kwargs,
        reward_config=reward_cfg,
        device=args.device,
        disable_omp=True,
        num_envs=max(1, args.num_envs),
        log_type=args.log_type,
        ppo_n_steps=max(32, args.ppo_n_steps),
        ppo_batch_size=max(64, args.ppo_batch_size),
        ppo_epochs=max(1, args.ppo_epochs),
        ppo_lr=float(args.ppo_lr),
        ppo_ent_coef=args.ppo_ent_coef,
        explorer_pop_scale=max(0.1, args.explorer_pop_scale),
        exploiter_pop_scale=max(0.1, args.exploiter_pop_scale),
        progress_bar=bool(args.progress_bar),
    )
