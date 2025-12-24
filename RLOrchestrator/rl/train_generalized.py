"""
Generalized training script for the RL Orchestrator.

This script trains a single policy across multiple problems and solver combinations.
It supports randomized solver pairings per episode, enabling the agent to learn
universal timing signals that generalize across different problem/solver contexts.

Supported Problems:
- TSP (7 explorers x 6 exploiters = 42 pairings)
- MaxCut (7 explorers x 7 exploiters = 49 pairings)  
- Knapsack (7 explorers x 7 exploiters = 49 pairings)
- NKL (11 explorers x 13 exploiters = 143 pairings)
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from RLOrchestrator.core.env_factory import create_env
from RLOrchestrator.core.utils import parse_int_range, setup_logging
from RLOrchestrator.problems.registry import instantiate_problem, list_problem_definitions
from RLOrchestrator.rl.callbacks import PeriodicBestCheckpoint
from RLOrchestrator.rl.training.run_artifacts import prepare_run_artifacts
from RLOrchestrator.rl.training.runner import build_vec_env, load_or_create_ppo, choose_vec_env_type
from RLOrchestrator.rl.training.cli_args import (
    add_budget_args,
    add_full_ppo_args,
    add_model_io_args,
    add_training_core_args,
    add_vec_env_args,
)


def make_env_fn(
    problem_names: List[str],
    rank: int,
    max_decision_spec,
    search_step_spec,
    max_search_steps: Optional[int],
    reward_clip: float,
    seed: int,
    session_id: int,
    log_dir: str,
):
    """
    Factory function for multiprocessed env.
    
    Each reset randomly selects:
    1. A problem from problem_names
    2. A random explorer/exploiter pair from that problem's solver pool
    """
    def _init():
        env_seed = seed + rank

        def episode_factory(reset_seed: int | None):
            # Sample a new (problem, solver pair) each episode.
            chosen = random.choice(problem_names)
            bundle = instantiate_problem(chosen)
            stage_map = _stage_map(bundle.stages)
            return bundle.problem, stage_map["exploration"], stage_map["exploitation"]

        # Build an initial bundle once; subsequent resets resample via episode_factory.
        bundle = instantiate_problem(random.choice(problem_names))
        stage_map = _stage_map(bundle.stages)
        
        env = create_env(
            problem=bundle.problem,
            exploration_solver=stage_map["exploration"],
            exploitation_solver=stage_map["exploitation"],
            max_decision_steps=max_decision_spec,
            search_steps_per_decision=search_step_spec,
            max_search_steps=max_search_steps,
            reward_clip=reward_clip,
            session_id=session_id,
            log_type='train_generalized',
            log_dir=log_dir,
            episode_factory=episode_factory,
            emit_init_summary=(rank == 0),
        )
        env.reset(seed=env_seed)
        return env
    return _init


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generalized training across multiple problems with solver-agnostic design."
    )
    g_train = parser.add_argument_group("Training")
    g_env = parser.add_argument_group("Environment")
    g_problem = parser.add_argument_group("Problem")
    g_ppo = parser.add_argument_group("PPO")
    g_io = parser.add_argument_group("Model I/O")

    add_training_core_args(g_train, total_timesteps_default=250000)
    add_vec_env_args(g_env, num_envs_default=4, vec_env_default="auto")
    add_budget_args(g_env, max_decisions_default="200", search_steps_per_decision_default="10")

    g_problem.add_argument(
        "--problems",
        type=str,
        default="all",
        help="Comma-separated problem names or 'all' for all registered problems.",
    )
    g_problem.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    add_full_ppo_args(g_ppo)
    add_model_io_args(g_io, model_output_default="results/models/ppo_generalized")
    return parser


def main():
    args = build_parser().parse_args()
    
    # Setup threading for performance
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    session_id = int(time.time())
    artifacts = prepare_run_artifacts(
        mode="train_generalized",
        problem="multi",
        model_output=args.model_output,
        session_id=session_id,
        args=vars(args),
    )
    logger = setup_logging(
        'train_generalized',
        'multi',
        log_dir=str(artifacts.logs_dir),
        session_id=session_id,
    )

    # Get problem list
    definitions = list_problem_definitions()
    if args.problems == "all":
        problem_names = list(definitions.keys())
    else:
        problem_names = [p.strip() for p in args.problems.split(",")]
        invalid = set(problem_names) - set(definitions.keys())
        if invalid:
            raise ValueError(f"Unknown problems: {invalid}. Available: {list(definitions.keys())}")
    
    logger.info(f"Starting generalized training with problems: {problem_names}")
    logger.info(f"Args: {args}")
    
    # Parse parameter ranges
    max_decision_spec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec = parse_int_range(args.search_steps_per_decision, min_value=1, label="search-steps-per-decision")
    
    # Build environment factory
    env_kwargs = dict(
        problem_names=problem_names,
        max_decision_spec=max_decision_spec,
        search_step_spec=search_step_spec,
        max_search_steps=args.max_search_steps,
        reward_clip=args.reward_clip,
        seed=args.seed,
        session_id=session_id,
        log_dir=str(artifacts.logs_dir),
    )
    
    num_envs = max(1, args.num_envs)
    
    if num_envs == 1:
        env_fns = [make_env_fn(rank=0, **env_kwargs)]
    else:
        env_fns = [make_env_fn(rank=i, **env_kwargs) for i in range(num_envs)]

    vec_type = choose_vec_env_type(vec_env=args.vec_env, num_envs=num_envs)
    env = build_vec_env(env_fns, num_envs=num_envs, vec_env_type=vec_type, single_env_mode="dummy")
    
    # Setup output path
    output_path = artifacts.final_model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(
        f"Run: mode=train_generalized, session_id={session_id}, "
        f"problems={len(problem_names)}, total_timesteps={args.total_timesteps}, "
        f"num_envs={num_envs}"
    )
    
    # Load or create model
    checkpoint_path = Path(args.load_model).expanduser() if args.load_model else None
    model, reset_flag = load_or_create_ppo(
        checkpoint_path=checkpoint_path,
        env=env,
        create_kwargs={
            "policy": "MlpPolicy",
            "verbose": 0,
            "device": args.device,
            "learning_rate": args.ppo_learning_rate,
            "n_steps": args.ppo_n_steps,
            "batch_size": args.ppo_batch_size,
            "n_epochs": args.ppo_epochs,
            "ent_coef": args.ppo_ent_coef,
            "gamma": args.ppo_gamma,
            "gae_lambda": args.ppo_gae_lambda,
            "policy_kwargs": {"net_arch": [64, 64]},
        },
    )
    if reset_flag:
        logger.info(f"Created new PPO model with learning rate: {args.ppo_learning_rate}")
    else:
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Setup callbacks
    callbacks = [
        PeriodicBestCheckpoint(
            total_timesteps=args.total_timesteps,
            save_dir=artifacts.checkpoints_dir,
            save_prefix=output_path.stem,
            verbose=0,
            log_episodes=True,
            logger=logger,
        )
    ]
    callback = CallbackList(callbacks)
    
    # Train
    logger.info("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=reset_flag,
        callback=callback,
        progress_bar=args.progress_bar,
    )
    
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")
    
    env.close()


def _stage_map(stages):
    """Convert stage bindings to dict for easy lookup."""
    mapping = {binding.name: binding.solver for binding in stages}
    missing = {"exploration", "exploitation"} - mapping.keys()
    if missing:
        raise ValueError(f"Problem bundle missing stages: {sorted(missing)}")
    return mapping


if __name__ == "__main__":
    main()
