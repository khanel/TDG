"""
NKL-specific PPO training script with solver-agnostic design.

This script trains a PPO agent to orchestrate a 3-phase optimization pipeline:
- Phase 1 (Exploration): Randomly selected from 11 explorer variants
- Phase 2 (Exploitation): Randomly selected from 13 exploiter variants
- Phase 3 (Termination): Episode ends, quality evaluated

Solver-Agnostic Training:
-------------------------
The agent learns UNIVERSAL timing signals (when to switch phases) that work
across ANY explorer/exploiter combination. Each episode randomly samples:
- 1 of 11 explorers (MAP-Elites, GWO, PSO, GA, ABC, WOA, HHO, MPA, SMA, GSA, Diversity)
- 1 of 13 exploiters (BinaryPSO, GWO, PSO, GA, L-SHADE, WOA, HHO, MPA, SMA, GSA, HillClimbing, Memetic, ABC)

This results in 143 unique solver pairings for robust policy learning.

Reward Function: Effectiveness-First Reward (EFR)
- EFFECTIVENESS FIRST: Quality gates everything (threshold = 0.7)
- EFFICIENCY SECOND: Budget savings only count if quality is good

Action semantics are UNIFIED across all phases:
- Action 0: STAY in current phase
- Action 1: ADVANCE to next phase
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ...core.env_factory import create_env
from ...core.utils import parse_int_range, setup_logging
from ...problems.registry import instantiate_problem
from ...rl.callbacks import PeriodicBestCheckpoint
from ...rl.training.run_artifacts import prepare_run_artifacts
from ...rl.training.runner import build_vec_env, load_or_create_ppo, choose_vec_env_type
from ...rl.training.cli_args import (
    add_budget_args,
    add_full_ppo_args,
    add_model_io_args,
    add_training_core_args,
    add_vec_env_args,
)


def make_env_fn(
    rank: int,
    n_items_range,
    k_interactions_range,
    max_decision_spec,
    search_step_spec,
    max_search_steps: Optional[int],
    reward_clip: float,
    seed: Optional[int],
    session_id: int,
    log_dir: str,
):
    """
    Factory function for creating environments.
    
    Each environment independently samples a random explorer/exploiter pair
    from the 11x13 solver pool on every reset, enabling solver-agnostic learning.
    """
    def _init():
        env_seed = seed + rank if seed is not None else None
        adapter_kwargs = {
            "n_items": n_items_range,
            "k_interactions": k_interactions_range,
            "seed": env_seed,
        }

        def episode_factory(reset_seed: int | None):
            # No solver_kwargs override: let registry randomly pick from full pool
            episode_seed = env_seed if reset_seed is None else reset_seed
            bundle = instantiate_problem(
                "nkl",
                adapter_kwargs={
                    **adapter_kwargs,
                    "seed": episode_seed,
                },
            )
            stage_map = _stage_map(bundle.stages)
            return bundle.problem, stage_map["exploration"], stage_map["exploitation"]

        # Build the initial bundle once; subsequent resets resample via episode_factory.
        bundle = instantiate_problem("nkl", adapter_kwargs=adapter_kwargs)
        stage_map = _stage_map(bundle.stages)
        exploration = stage_map["exploration"]
        exploitation = stage_map["exploitation"]
        
        env = create_env(
            bundle.problem,
            exploration,
            exploitation,
            max_decision_steps=max_decision_spec,
            search_steps_per_decision=search_step_spec,
            max_search_steps=max_search_steps,
            reward_clip=reward_clip,
            log_type='train',
            log_dir=log_dir,
            session_id=session_id,
            emit_init_summary=(rank == 0),
            episode_factory=episode_factory,
        )
        if env_seed is not None:
            env.reset(seed=env_seed)

        return env

    return _init


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NKL PPO Training with Solver-Agnostic Design (11x13=143 pairings)"
    )

    g_train = parser.add_argument_group("Training")
    g_env = parser.add_argument_group("Environment")
    g_ppo = parser.add_argument_group("PPO")
    g_io = parser.add_argument_group("Model I/O")
    g_problem = parser.add_argument_group("Problem")

    add_training_core_args(g_train, total_timesteps_default=100000)
    add_vec_env_args(g_env, num_envs_default=4, vec_env_default="auto")
    add_full_ppo_args(g_ppo)
    add_budget_args(g_env, max_decisions_default="200", search_steps_per_decision_default="10")
    add_model_io_args(g_io, model_output_default="results/models/ppo_nkl_solveragnostic")

    g_problem.add_argument("--nkl-n-items", type=str, default="100")
    g_problem.add_argument("--nkl-k-interactions", type=str, default="5")
    g_problem.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    
    # Setup threading for performance
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    
    # Enable TF32 for CUDA performance
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    session_id = int(time.time())
    artifacts = prepare_run_artifacts(
        mode="train",
        problem="nkl",
        model_output=args.model_output,
        session_id=session_id,
        args=vars(args),
    )
    logger = setup_logging('train', 'nkl', log_dir=str(artifacts.logs_dir), session_id=session_id)
    
    logger.info(f"Starting NKL solver-agnostic training with args: {args}")

    # Parse parameter ranges
    n_items_range = parse_int_range(args.nkl_n_items, min_value=2, label="nkl-n-items")
    k_interactions_range = parse_int_range(args.nkl_k_interactions, min_value=0, label="nkl-k-interactions")
    max_decision_spec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec = parse_int_range(args.search_steps_per_decision, min_value=1, label="search-steps-per-decision")

    # Build environment factory
    env_kwargs = dict(
        n_items_range=n_items_range,
        k_interactions_range=k_interactions_range,
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

    # Logging: solver-agnostic training emphasizes diverse pairings
    logger.info(
        f"Run: mode=train, session_id={session_id}, problem=nkl, "
        f"total_timesteps={args.total_timesteps}, num_envs={num_envs}, "
        f"solver_pairings=143 (11 explorers x 13 exploiters)"
    )
    logger.info(
        f"Config: max_decisions={args.max_decisions}, "
        f"steps_per_decision={args.search_steps_per_decision}, "
        f"reward=EFR, lr={args.ppo_learning_rate}"
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

    # Save final model
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
