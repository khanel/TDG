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


def make_env_fn(
    problem_names: List[str],
    rank: int,
    max_decision_spec,
    search_step_spec,
    max_search_steps: Optional[int],
    reward_clip: float,
    seed: int,
    session_id: int,
):
    """
    Factory function for multiprocessed env.
    
    Each reset randomly selects:
    1. A problem from problem_names
    2. A random explorer/exploiter pair from that problem's solver pool
    """
    def _init():
        env_seed = seed + rank
        problem_name = random.choice(problem_names)
        bundle = instantiate_problem(problem_name)
        stage_map = _stage_map(bundle.stages)
        
        for solver in stage_map.values():
            if hasattr(solver, "initialize"):
                solver.initialize()
        
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
            emit_init_summary=(rank == 0),
        )
        env.reset(seed=env_seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Generalized training across multiple problems with solver-agnostic design."
    )
    
    # === Training Configuration ===
    parser.add_argument("--total-timesteps", type=int, default=250000,
                        help="Total timesteps for training.")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments.")
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto",
                        help="Vectorized environment type.")
    
    # === Problem Selection ===
    parser.add_argument("--problems", type=str, default="all",
                        help="Comma-separated problem names or 'all' for all registered problems.")
    
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
    
    # === Model I/O ===
    parser.add_argument("--model-output", type=str, default="results/models/ppo_generalized",
                        help="Path to save the trained model.")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load a checkpoint for continued training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    
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
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    session_id = int(time.time())
    logger = setup_logging('train_generalized', 'multi', session_id=session_id)

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
    )
    
    num_envs = max(1, args.num_envs)
    
    if num_envs == 1:
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
    
    logger.info(
        f"Run: mode=train_generalized, session_id={session_id}, "
        f"problems={len(problem_names)}, total_timesteps={args.total_timesteps}, "
        f"num_envs={num_envs}"
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
