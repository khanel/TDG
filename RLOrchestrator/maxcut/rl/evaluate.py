"""
Evaluation script for trained Max-Cut orchestrator policies.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ...core.orchestrator import Orchestrator
from ...rl.environment import RLEnvironment
from ...maxcut.adapter import MaxCutAdapter
from ...maxcut.solvers import MaxCutRandomExplorer, MaxCutLocalSearch


def _load_weights(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Weight matrix not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        data = np.load(file_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "arr_0" in data:
                return np.asarray(data["arr_0"], dtype=float)
            raise ValueError(f"NPZ file {file_path} must contain array 'arr_0'")
        return np.asarray(data, dtype=float)
    return np.loadtxt(file_path, dtype=float)


def _plot_partition(weights: np.ndarray, mask: np.ndarray, save_path: Path) -> None:
    from networkx import Graph, draw_networkx, spring_layout

    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = weights.shape[0]
    g = Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if weights[i, j] > 0:
                g.add_edge(i, j)
    pos = spring_layout(g, seed=42)
    colors = ["tab:blue" if bit == 0 else "tab:orange" for bit in mask]
    plt.figure(figsize=(6, 6))
    draw_networkx(g, pos=pos, node_color=colors, with_labels=True, node_size=300, edge_color="grey")
    plt.title("Max-Cut partition")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ppo_maxcut.zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--max-decisions", type=int, default=200)
    parser.add_argument("--search-steps-per-decision", type=int, default=1)
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--n-nodes", type=int, default=64)
    parser.add_argument("--edge-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ensure-connected", action="store_true", default=False)
    parser.add_argument("--weights-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs/maxcut")
    parser.add_argument("--partition-image", type=str, default="partition.png")
    parser.add_argument("--fitness-image", type=str, default="fitness.png")
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    weights = _load_weights(args.weights_file)
    problem = MaxCutAdapter(
        weight_matrix=weights,
        n_nodes=args.n_nodes,
        edge_probability=args.edge_probability,
        seed=args.seed,
        ensure_connected=args.ensure_connected,
    )

    exploration = MaxCutRandomExplorer(
        problem,
        population_size=64,
        flip_probability=0.15,
        elite_fraction=0.25,
        seed=args.seed,
    )
    exploitation = MaxCutLocalSearch(
        problem,
        population_size=16,
        moves_per_step=8,
        escape_probability=0.05,
        seed=args.seed,
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
        weights_snapshot = np.asarray(env.orchestrator.problem.maxcut_problem.weights, dtype=float).copy()
        episodes_info.append({
            "index": episode_idx,
            "solution": episode_best_solution,
            "fitness": episode_best_fitness,
            "weights": weights_snapshot,
            "steps": episode_steps.copy(),
            "history": episode_fitness.copy(),
            "switch_steps": episode_switch_steps.copy(),
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
    partition_base = Path(args.partition_image)
    partition_ext = partition_base.suffix or ".png"
    partition_stem = partition_base.stem if partition_base.suffix else partition_base.name
    fitness_base = Path(args.fitness_image)
    fitness_ext = fitness_base.suffix or ".png"
    fitness_stem = fitness_base.stem if fitness_base.suffix else fitness_base.name

    best_episode = min(episodes_info, key=lambda d: d["fitness"] if d["fitness"] is not None else float("inf"))
    print(f"Best episode #{best_episode['index']} achieved fitness {best_episode['fitness']:.3f}")

    warned_networkx = False
    for info in episodes_info:
        idx = info["index"]
        solution = info["solution"]
        if solution is None:
            continue
        mask = np.asarray(solution.representation, dtype=int)
        partition_path = output_dir / f"{partition_stem}_ep{idx}{partition_ext}"
        try:
            _plot_partition(info["weights"], mask, partition_path)
        except ImportError:
            if not warned_networkx:
                print("networkx not available; skipping partition plots.")
                warned_networkx = True

        steps = info["steps"]
        history = info["history"]
        if steps and history:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, [-fit for fit in history], marker="o", linewidth=1.5)
            for switch_idx, step in enumerate(info["switch_steps"]):
                ax.axvline(step, color="red", linestyle="--", alpha=0.6, label="Phase switch" if switch_idx == 0 else "")
            if info["switch_steps"]:
                ax.legend()
            ax.set_title(f"Cut value over decision steps (episode {idx})")
            ax.set_xlabel("Decision step")
            ax.set_ylabel("Total cut value")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(left=0, right=max(steps[-1] + 2, len(steps)))
            fig.tight_layout()
            fitness_path = output_dir / f"{fitness_stem}_ep{idx}{fitness_ext}"
            fig.savefig(fitness_path, dpi=150)
            plt.close(fig)

    print(f"Saved episode plots to {output_dir}")


if __name__ == "__main__":
    main()
