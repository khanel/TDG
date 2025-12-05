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
        # No solver_kwargs override: let registry randomly pick from full pool
        bundle = instantiate_problem(
            "nkl",
            adapter_kwargs=adapter_kwargs,
        )
        stage_map = _stage_map(bundle.stages)
        exploration = stage_map["exploration"]
        exploitation = stage_map["exploitation"]
        
        # Initialize solvers
        for solver in stage_map.values():
            if hasattr(solver, "initialize"):
                solver.initialize()
        
        env = create_env(
            bundle.problem,
            exploration,
            exploitation,
            max_decision_steps=max_decision_spec,
            search_steps_per_decision=search_step_spec,
            max_search_steps=max_search_steps,
            reward_clip=reward_clip,
            log_type='train',
            log_dir='logs',
            session_id=session_id,
            emit_init_summary=(rank == 0),
        )
        if env_seed is not None:
            env.reset(seed=env_seed)

        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="NKL PPO Training with Solver-Agnostic Design (11x13=143 pairings)"
    )
    
    # === Training Configuration ===
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total timesteps for training.")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments.")
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto",
                        help="Vectorized environment type.")
    
    # === PPO Hyperparameters ===
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4,
                        help="PPO learning rate.")
    parser.add_argument("--ppo-n-steps", type=int, default=2048,
                        help="Number of steps per environment per update.")
    parser.add_argument("--ppo-batch-size", type=int, default=64,
                        help="PPO minibatch size.")
    parser.add_argument("--ppo-epochs", type=int, default=10,
                        help="Number of epochs when optimizing surrogate loss.")
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01,
                        help="Entropy coefficient for exploration bonus.")
    parser.add_argument("--ppo-gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95,
                        help="GAE lambda for advantage estimation.")
    
    # === Environment Configuration ===
    parser.add_argument("--max-decisions", type=str, default="200",
                        help="Max agent decision steps per episode (int or 'min-max').")
    parser.add_argument("--search-steps-per-decision", type=str, default="10",
                        help="Solver iterations per agent decision (int or 'min-max').")
    parser.add_argument("--max-search-steps", type=int, default=None,
                        help="Optional hard cap on total solver iterations.")
    parser.add_argument("--reward-clip", type=float, default=1.0,
                        help="Clip reward magnitude.")
    
    # === NKL Problem Parameters ===
    parser.add_argument("--nkl-n-items", type=str, default="100",
                        help="NKL N: problem dimension (int or 'min-max' for range).")
    parser.add_argument("--nkl-k-interactions", type=str, default="5",
                        help="NKL K: epistatic interactions (int or 'min-max' for range).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    
    # === Model I/O ===
    parser.add_argument("--model-output", type=str, default="results/models/ppo_nkl_solveragnostic",
                        help="Path to save the trained model.")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load a checkpoint for continued training.")
    
    # === Hardware ===
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cpu, cuda, auto).")
    parser.add_argument("--progress-bar", action="store_true", default=False,
                        help="Show training progress bar.")
    
    args = parser.parse_args()
    
    # Setup threading for performance
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    
    # Enable TF32 for CUDA performance
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    session_id = int(time.time())
    logger = setup_logging('train', 'nkl', session_id=session_id)
    
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
    )
    
    num_envs = max(1, args.num_envs)
    
    if num_envs == 1:
        # Single environment (no vectorization overhead)
        env = DummyVecEnv([make_env_fn(rank=0, **env_kwargs)])
    else:
        vec_type = args.vec_env
        if vec_type == "auto":
            vec_type = "subproc" if num_envs > 1 else "dummy"
        
        env_fns = [make_env_fn(rank=i, **env_kwargs) for i in range(num_envs)]
        if vec_type == "subproc":
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)

    # Setup output path
    output_path = Path(args.model_output)
    if output_path.suffix != ".zip":
        output_path = output_path.with_suffix(".zip")
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
    reset_flag = True
    
    if checkpoint_path and checkpoint_path.exists():
        model = PPO.load(checkpoint_path, env=env)
        reset_flag = False
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device=args.device,
            learning_rate=args.ppo_learning_rate,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            n_epochs=args.ppo_epochs,
            ent_coef=args.ppo_ent_coef,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
            policy_kwargs={"net_arch": [64, 64]},
        )
        logger.info(f"Created new PPO model with learning rate: {args.ppo_learning_rate}")

    # Setup callbacks
    callbacks = [
        PeriodicBestCheckpoint(
            total_timesteps=args.total_timesteps,
            save_dir=output_path.parent,
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
