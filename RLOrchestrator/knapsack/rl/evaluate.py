"""Evaluation script for Knapsack orchestrator policies."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ...core.orchestrator import Orchestrator
from ...core.utils import parse_int_range
from ...rl.environment import RLEnvironment
from ...knapsack.adapter import KnapsackAdapter
from ...knapsack.solvers import KnapsackRandomExplorer, KnapsackLocalSearch


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


def _plot_selection(values: np.ndarray, weights: np.ndarray, mask: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(values.size)
    selected = mask.astype(bool)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(idx, values, color=np.where(selected, "tab:orange", "tab:blue"), alpha=0.7)
    ax1.set_ylabel("Value")
    ax1.set_xlabel("Item index")
    ax1.set_title("Knapsack selection")
    ax2 = ax1.twinx()
    ax2.plot(idx, weights, color="tab:green", linestyle="--", label="Weight")
    ax2.set_ylabel("Weight")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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
    parser.add_argument("--model-path", type=str, default="ppo_knapsack.zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--max-decisions", type=str, default="200")
    parser.add_argument("--search-steps-per-decision", type=str, default="1")
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--values-file", type=str, default=None)
    parser.add_argument("--weights-file", type=str, default=None)
    parser.add_argument("--capacity", type=float, default=None)
    parser.add_argument("--n-items", type=str, default="50")
    parser.add_argument("--value-range", type=float, nargs=2, default=(1.0, 100.0))
    parser.add_argument("--weight-range", type=float, nargs=2, default=(1.0, 50.0))
    parser.add_argument("--capacity-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs/knapsack")
    parser.add_argument("--selection-image", type=str, default="selection.png")
    parser.add_argument("--fitness-image", type=str, default="fitness.png")
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

    items_range = _parse_items_spec(args.n_items)

    problem = KnapsackAdapter(
        values=values.tolist() if values is not None else None,
        weights=weights.tolist() if weights is not None else None,
        capacity=capacity,
        n_items=items_range,
        value_range=tuple(args.value_range),
        weight_range=tuple(args.weight_range),
        capacity_ratio=args.capacity_ratio,
        seed=args.seed,
    )

    exploration = KnapsackRandomExplorer(problem, population_size=10, flip_probability=0.15, elite_fraction=0.33, seed=args.seed)
    exploitation = KnapsackLocalSearch(problem, population_size=6, moves_per_step=6, escape_probability=0.05, seed=args.seed)
    for solver in (exploration, exploitation):
        if hasattr(solver, "initialize"):
            solver.initialize()

    orchestrator = Orchestrator(problem, exploration, exploitation, start_phase="exploration")
    orchestrator._update_best()
    max_decision_spec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec = parse_int_range(args.search_steps_per_decision, min_value=1, label="search-steps-per-decision")
    env = RLEnvironment(
        orchestrator,
        max_decision_steps=max_decision_spec,
        search_steps_per_decision=search_step_spec,
        max_search_steps=args.max_search_steps,
        reward_clip=args.reward_clip,
    )

    model = PPO.load(args.model_path)

    episodes_info: list[dict] = []
    returns: list[float] = []

    for episode_idx in range(1, max(1, args.episodes) + 1):
        obs, _ = env.reset()
        done = False
        step_idx = 0
        ep_return = 0.0
        episode_steps: list[int] = []
        episode_fitness: list[float] = []
        episode_switch_steps: list[int] = []
        episode_best_solution = None
        episode_best_fitness = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            if int(action) == 1 and env.orchestrator.get_phase() == "exploration":
                episode_switch_steps.append(step_idx)
            obs, reward, done, _, _ = env.step(int(action))
            ep_return += reward
            candidate = env.orchestrator.get_best_solution()
            if candidate and candidate.fitness is not None:
                episode_steps.append(step_idx)
                episode_fitness.append(candidate.fitness)
                if candidate.fitness < episode_best_fitness:
                    episode_best_solution = candidate.copy()
                    episode_best_fitness = candidate.fitness
            step_idx += 1
        returns.append(ep_return)

        if episode_best_solution is None:
            episode_best_solution = env.orchestrator.get_best_solution().copy()
            episode_best_fitness = episode_best_solution.fitness if episode_best_solution else float("inf")
        values_arr = np.asarray(env.orchestrator.problem.knapsack_problem.values, dtype=float).copy()
        weights_arr = np.asarray(env.orchestrator.problem.knapsack_problem.weights, dtype=float).copy()
        episodes_info.append({
            "index": episode_idx,
            "solution": episode_best_solution,
            "fitness": episode_best_fitness,
            "values": values_arr,
            "weights": weights_arr,
            "steps": episode_steps.copy(),
            "history": episode_fitness.copy(),
            "switch_steps": episode_switch_steps.copy(),
            "capacity": float(env.orchestrator.problem.knapsack_problem.capacity),
        })

    env.close()

    if not episodes_info:
        print("No solution discovered during evaluation.")
        return

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if len(returns) > 1 else 0.0
    print(f"Evaluated {len(returns)} episode(s) | mean return: {mean_return:.3f} ± {std_return:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_base = Path(args.selection_image)
    selection_ext = selection_base.suffix or ".png"
    selection_stem = selection_base.stem if selection_base.suffix else selection_base.name
    fitness_base = Path(args.fitness_image)
    fitness_ext = fitness_base.suffix or ".png"
    fitness_stem = fitness_base.stem if fitness_base.suffix else fitness_base.name

    best_episode = min(episodes_info, key=lambda d: d["fitness"] if d["fitness"] is not None else float("inf"))
    print(f"Best episode #{best_episode['index']} achieved fitness {best_episode['fitness']:.3f}")

    for info in episodes_info:
        idx = info["index"]
        solution = info["solution"]
        if solution is None:
            continue
        selection_path = output_dir / f"{selection_stem}_ep{idx}{selection_ext}"
        mask = np.asarray(solution.representation, dtype=int)
        _plot_selection(info["values"], info["weights"], mask, selection_path)

        steps = info["steps"]
        history = info["history"]
        if steps and history:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, [-fit for fit in history], marker="o", linewidth=1.5)
            for switch_idx, step in enumerate(info["switch_steps"]):
                ax.axvline(step, color="red", linestyle="--", alpha=0.6, label="Phase switch" if switch_idx == 0 else "")
            if info["switch_steps"]:
                ax.legend()
            ax.set_title(f"Knapsack value over decision steps (episode {idx})")
            ax.set_xlabel("Decision step")
            ax.set_ylabel("Total value (higher is better)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(left=0, right=max(steps[-1] + 2, len(steps)))
            fitness_path = output_dir / f"{fitness_stem}_ep{idx}{fitness_ext}"
            fig.tight_layout()
            fig.savefig(fitness_path, dpi=150)
            plt.close(fig)

    print(f"Saved episode plots to {output_dir}")


if __name__ == "__main__":
    main()
