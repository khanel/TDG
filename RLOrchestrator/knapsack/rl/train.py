"""Knapsack PPO training script."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ...core.orchestrator import Orchestrator
from ...rl.environment import RLEnvironment
from ...knapsack.adapter import KnapsackAdapter
from ...knapsack.solvers import KnapsackRandomExplorer, KnapsackLocalSearch
from ...rl.callbacks import PeriodicBestCheckpoint


def _load_array(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Knapsack data file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        data = np.load(file_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "arr_0" in data:
                return np.asarray(data["arr_0"], dtype=float)
            raise ValueError(f"NPZ file {file_path} must contain array 'arr_0'")
        return np.asarray(data, dtype=float)
    return np.loadtxt(file_path, dtype=float)


def _parse_items_spec(value: str) -> Tuple[int, int]:
    text = str(value).strip()
    for sep in ("-", ":", ","):
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) == 2:
                lo = int(float(parts[0]))
                hi = int(float(parts[1]))
                lo, hi = sorted((lo, hi))
                return (max(1, lo), max(1, hi))
    num = int(float(text))
    num = max(1, num)
    return (num, num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--exploration-population", type=int, default=10)
    parser.add_argument("--exploitation-population", type=int, default=6)
    parser.add_argument("--max-decisions", type=int, default=200)
    parser.add_argument("--search-steps-per-decision", type=int, default=1)
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--progress-bar", action="store_true", default=False)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--model-output", type=str, default="ppo_knapsack")
    parser.add_argument("--values-file", type=str, default=None)
    parser.add_argument("--weights-file", type=str, default=None)
    parser.add_argument("--capacity", type=float, default=None)
    parser.add_argument("--n-items", type=str, default="50")
    parser.add_argument("--value-range", type=float, nargs=2, default=(1.0, 100.0))
    parser.add_argument("--weight-range", type=float, nargs=2, default=(1.0, 50.0))
    parser.add_argument("--capacity-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    values = _load_array(args.values_file)
    weights = _load_array(args.weights_file)
    capacity = args.capacity

    if values is not None and weights is None or weights is not None and values is None:
        raise ValueError("Both values and weights must be provided together.")
    if values is not None and weights is not None:
        if values.shape != weights.shape:
            raise ValueError("values and weights must have matching shape")
        if capacity is None:
            raise ValueError("capacity must be provided when supplying custom values/weights")

    def make_env_fn(rank: int):
        def _init():
            seed = args.seed + rank if args.seed is not None else None
    items_range = _parse_items_spec(args.n_items)

    def make_env_fn(rank: int):
        def _init():
            seed = args.seed + rank if args.seed is not None else None
            problem = KnapsackAdapter(
                values=values.tolist() if values is not None else None,
                weights=weights.tolist() if weights is not None else None,
                capacity=capacity,
                n_items=items_range,
                value_range=tuple(args.value_range),
                weight_range=tuple(args.weight_range),
                capacity_ratio=args.capacity_ratio,
                seed=seed,
            )
            exploration = KnapsackRandomExplorer(
                problem,
                population_size=max(1, args.exploration_population),
                flip_probability=0.15,
                elite_fraction=0.33,
                seed=seed,
            )
            exploitation = KnapsackLocalSearch(
                problem,
                population_size=max(1, args.exploitation_population),
                moves_per_step=6,
                escape_probability=0.05,
                seed=seed,
            )
            for solver in (exploration, exploitation):
                if hasattr(solver, "initialize"):
                    solver.initialize()
            orchestrator = Orchestrator(problem, exploration, exploitation, start_phase="exploration")
            orchestrator._update_best()
            env = RLEnvironment(
                orchestrator,
                max_decision_steps=args.max_decisions,
                search_steps_per_decision=args.search_steps_per_decision,
                max_search_steps=args.max_search_steps,
                reward_clip=args.reward_clip,
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

    checkpoint_path = Path(args.load_model).expanduser() if args.load_model else None
    if checkpoint_path and checkpoint_path.exists():
        model = PPO.load(checkpoint_path, env=env)
        reset_flag = False
    else:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=args.ppo_learning_rate)
        reset_flag = True

    callbacks = []
    if args.progress_bar:
        callbacks.append(ProgressBarCallback())
    callbacks.append(
        PeriodicBestCheckpoint(
            total_timesteps=args.total_timesteps,
            save_dir=output_path.parent,
            save_prefix=output_path.stem,
            verbose=1,
            log_episodes=True,
        )
    )
    callback = CallbackList(callbacks)

    model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=reset_flag, callback=callback)

    model.save(output_path)
    env.close()


if __name__ == "__main__":
    main()
