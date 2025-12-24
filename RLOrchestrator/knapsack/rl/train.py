"""
Knapsack-specific PPO training script.
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ...core.env_factory import create_env
from ...core.utils import parse_int_range, parse_float_range, setup_logging
from ...problems.registry import instantiate_problem
from ...rl.callbacks import PeriodicBestCheckpoint
from ...rl.training.run_artifacts import prepare_run_artifacts
from ...rl.training.runner import build_vec_env, load_or_create_ppo, choose_vec_env_type
from ...rl.training.cli_args import (
    add_basic_ppo_args,
    add_budget_args,
    add_model_io_args,
    add_training_core_args,
    add_vec_env_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    g_train = parser.add_argument_group("Training")
    g_env = parser.add_argument_group("Environment")
    g_ppo = parser.add_argument_group("PPO")
    g_io = parser.add_argument_group("Model I/O")
    g_problem = parser.add_argument_group("Problem")

    add_training_core_args(g_train, total_timesteps_default=100000)
    g_train.add_argument("--exploration-population", type=int, default=64)
    g_train.add_argument("--exploitation-population", type=int, default=16)

    add_budget_args(g_env, max_decisions_default="200", search_steps_per_decision_default="10")
    add_vec_env_args(g_env, num_envs_default=1, vec_env_default="auto")

    add_basic_ppo_args(g_ppo, learning_rate_default=3e-4)

    add_model_io_args(g_io, model_output_default="ppo_knapsack")

    g_problem.add_argument("--knapsack-num-items", type=str, default="50")
    g_problem.add_argument("--knapsack-value-range", type=str, default="1.0-100.0")
    g_problem.add_argument("--knapsack-weight-range", type=str, default="1.0-50.0")
    g_problem.add_argument("--knapsack-capacity-ratio", type=float, default=0.5)
    g_problem.add_argument("--knapsack-seed", type=int, default=42)
    return parser

def main():
    session_id = int(time.time())
    args = build_parser().parse_args()

    artifacts = prepare_run_artifacts(
        mode="train",
        problem="knapsack",
        model_output=args.model_output,
        session_id=session_id,
        args=vars(args),
    )
    logger = setup_logging('train', 'knapsack', log_dir=str(artifacts.logs_dir), session_id=session_id)

    logger.info(f"Starting Knapsack training with args: {args}")

    num_items_range = parse_int_range(args.knapsack_num_items, min_value=1, label="knapsack-num-items")
    value_range = parse_float_range(args.knapsack_value_range, label="knapsack-value-range")
    weight_range = parse_float_range(args.knapsack_weight_range, label="knapsack-weight-range")

    max_decision_spec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec = parse_int_range(args.search_steps_per_decision, min_value=1, label="search-steps-per-decision")

    def make_env_fn(rank: int):
        def _init():
            seed = args.knapsack_seed + rank if args.knapsack_seed is not None else None
            adapter_kwargs = {
                "n_items": num_items_range,
                "value_range": value_range,
                "weight_range": weight_range,
                "capacity_ratio": args.knapsack_capacity_ratio,
                "seed": seed,
            }
            solver_overrides = {
                "exploration": {
                    "population_size": max(1, args.exploration_population),
                    "flip_probability": 0.15,
                    "elite_fraction": 0.25,
                    "seed": seed,
                },
                "exploitation": {
                    "population_size": max(1, args.exploitation_population),
                    "moves_per_step": 8,
                    "escape_probability": 0.05,
                    "seed": seed,
                },
            }
            bundle = instantiate_problem(
                "knapsack",
                adapter_kwargs=adapter_kwargs,
                solver_kwargs=solver_overrides,
            )
            stage_map = _stage_map(bundle.stages)
            exploration = stage_map["exploration"]
            exploitation = stage_map["exploitation"]
            for solver in stage_map.values():
                if hasattr(solver, "initialize"):
                    solver.initialize()
            env = create_env(
                bundle.problem,
                exploration,
                exploitation,
                max_decision_steps=max_decision_spec,
                search_steps_per_decision=search_step_spec,
                max_search_steps=args.max_search_steps,
                reward_clip=args.reward_clip,
                log_type='train',
                log_dir=str(artifacts.logs_dir),
                session_id=session_id,
                emit_init_summary=(rank == 0),
            )
            if seed is not None:
                env.reset(seed=seed)

            return env

        return _init

    num_envs = max(1, int(args.num_envs))
    env_fns = [make_env_fn(rank) for rank in range(num_envs)]
    vec_type = choose_vec_env_type(vec_env=args.vec_env, num_envs=num_envs)
    env = build_vec_env(env_fns, num_envs=num_envs, vec_env_type=vec_type, single_env_mode="raw")

    output_path = artifacts.final_model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # High-level start header (single line)
    logger.info(
        f"Run: mode=train, session_id={session_id}, problem=knapsack, total_timesteps={int(args.total_timesteps)}, num_envs={int(num_envs)}, vec_env={'subproc' if num_envs > 1 else 'single'}"
    )
    # High-level config line (single line)
    logger.info(
        f"Config: max_decisions={args.max_decisions}, steps_per_decision={args.search_steps_per_decision}, reward_clip={args.reward_clip}, learning_rate={args.ppo_learning_rate}"
    )

    checkpoint_path = Path(args.load_model).expanduser() if args.load_model else None
    model, reset_flag = load_or_create_ppo(
        checkpoint_path=checkpoint_path,
        env=env,
        create_kwargs={
            "policy": "MlpPolicy",
            "device": "cpu",
            "learning_rate": args.ppo_learning_rate,
            "verbose": 0,
        },
    )
    if reset_flag:
        logger.info(f"Created new PPO model with learning rate: {args.ppo_learning_rate}")

    callbacks = []
    callbacks.append(
        PeriodicBestCheckpoint(
            total_timesteps=args.total_timesteps,
            save_dir=artifacts.checkpoints_dir,
            save_prefix=output_path.stem,
            verbose=0,
            log_episodes=True,
            logger=logger,
        )
    )
    callback = CallbackList(callbacks)

    model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=reset_flag, callback=callback, progress_bar=False)

    model.save(output_path)
    env.close()


def _stage_map(stages):
    mapping = {binding.name: binding.solver for binding in stages}
    missing = {"exploration", "exploitation"} - mapping.keys()
    if missing:
        raise ValueError(f"Problem bundle missing stages: {sorted(missing)}")
    return mapping


if __name__ == "__main__":
    main()
