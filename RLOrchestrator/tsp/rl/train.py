"""
TSP-specific PPO training script.
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ...core.orchestrator import Orchestrator
from ...core.env_factory import create_env
from ...rl.callbacks import PeriodicBestCheckpoint
from ...core.utils import parse_int_range, parse_float_range, setup_logging
from ...rl.callbacks import PeriodicBestCheckpoint

def main():
    session_id = int(time.time())
    logger = setup_logging('train', 'tsp', session_id=session_id)
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--exploration-population", type=int, default=32)
    parser.add_argument("--exploitation-population", type=int, default=8)
    parser.add_argument("--max-decisions", type=str, default="200")
    parser.add_argument("--search-steps-per-decision", type=str, default="1")
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--progress-bar", action="store_true", default=False)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--model-output", type=str, default="ppo_tsp")
    parser.add_argument("--tsp-num-cities", type=str, default="20")
    parser.add_argument("--tsp-grid-size", type=float, default=100.0)
    parser.add_argument("--tsp-seed", type=int, default=42)
    parser.add_argument("--tsp-coords-file", type=str, default=None)
    parser.add_argument("--tsp-distance-file", type=str, default=None)
    args = parser.parse_args()

    logger.info(f"Starting TSP training with args: {args}")

    coords_arr = _load_array(args.tsp_coords_file)
    dist_arr = _load_array(args.tsp_distance_file)
    num_cities_range = _parse_num_cities_spec(args.tsp_num_cities)

    max_decision_spec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec = parse_int_range(args.search_steps_per_decision, min_value=1, label="search-steps-per-decision")

    def make_env_fn(rank: int):
        def _init():
            from ...core.orchestrator import Orchestrator
            from ...core.env_factory import create_env
            from ...tsp.adapter import TSPAdapter
            from ...tsp.solvers import TSPMapElites, TSPParticleSwarm

            seed = args.tsp_seed + rank if args.tsp_seed is not None else None
            problem = TSPAdapter(
                num_cities=num_cities_range,
                grid_size=args.tsp_grid_size,
                seed=seed,
                coords=coords_arr.tolist() if coords_arr is not None else None,
                distance_matrix=dist_arr.tolist() if dist_arr is not None else None,
            )
            exploration = TSPMapElites(
                problem,
                population_size=max(1, args.exploration_population),
                bins_per_dim=(16, 16),
                random_injection_rate=0.15,
                seed=seed,
            )
            exploitation = TSPParticleSwarm(
                problem,
                population_size=max(1, args.exploitation_population),
                seed=seed,
            )
            for solver in (exploration, exploitation):
                if hasattr(solver, "initialize"):
                    solver.initialize()
            orchestrator = Orchestrator(problem, exploration, exploitation, start_phase="exploration")
            orchestrator._update_best()
            env = create_env(
                problem,
                exploration,
                exploitation,
                max_decision_steps=max_decision_spec,
                search_steps_per_decision=search_step_spec,
                max_search_steps=args.max_search_steps,
                reward_clip=args.reward_clip,
                log_type='train',
                log_dir='logs',
                session_id=session_id,
                emit_init_summary=(rank == 0),
            )
            if seed is not None:
                env.reset(seed=seed)

            return env

        return _init

    num_envs = max(1, int(args.num_envs))
    if num_envs == 1:
        env = make_env_fn(0)()
    else:
        vec_type = args.vec_env
        if vec_type == "auto":
            vec_type = "subproc"
        env_fns = [make_env_fn(rank) for rank in range(num_envs)]
        if vec_type == "subproc":
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)

    output_path = Path(args.model_output)
    if output_path.suffix != ".zip":
        output_path = output_path.with_suffix(".zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # High-level start header (single line)
    logger.info(
        f"Run: mode=train, session_id={session_id}, problem=tsp, total_timesteps={int(args.total_timesteps)}, num_envs={int(num_envs)}, vec_env={'subproc' if num_envs > 1 else 'single'}"
    )
    # High-level config line (single line)
    logger.info(
        f"Config: max_decisions={args.max_decisions}, steps_per_decision={args.search_steps_per_decision}, reward_clip={args.reward_clip}, learning_rate={args.ppo_learning_rate}"
    )

    checkpoint_path = Path(args.load_model).expanduser() if args.load_model else None
    if checkpoint_path and checkpoint_path.exists():
        model = PPO.load(checkpoint_path, env=env)
        reset_flag = False
    else:
        model = PPO("MlpPolicy",env, device='cpu', learning_rate=args.ppo_learning_rate, verbose=0)
        logger.info(f"Created new PPO model with learning rate: {args.ppo_learning_rate}")
        reset_flag = True

    callbacks = []
    callbacks.append(
        PeriodicBestCheckpoint(
            total_timesteps=args.total_timesteps,
            save_dir=output_path.parent,
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


if __name__ == "__main__":
    main()
