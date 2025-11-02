"""
Evaluation script for trained TSP orchestrator policies.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ...core.orchestrator import Orchestrator
from ...core.utils import parse_int_range, setup_logging
from ...rl.environment import RLEnvironment
from ...tsp.adapter import TSPAdapter
from ...tsp.solvers import TSPMapElites, TSPParticleSwarm
from ...rl.eval_logging import EvaluationLogger, StepRecord, EpisodeSummary, DEFAULT_OBS_NAMES


def _load_array(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"TSP configuration file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        data = np.load(file_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "arr_0" in data:
                arr = np.asarray(data["arr_0"], dtype=float)
            else:
                raise ValueError(f"NPZ file {file_path} must contain array 'arr_0'")
        else:
            arr = np.asarray(data, dtype=float)
    else:
        arr = np.loadtxt(file_path, dtype=float)
    return arr


def _plot_tour(coords: np.ndarray, tour: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tour_idx = (tour - 1).astype(int)
    route = coords[tour_idx]
    closed = np.vstack([route, route[0]])
    plt.figure(figsize=(6, 6))
    plt.plot(closed[:, 0], closed[:, 1], "-o", markersize=4)
    for i, (x, y) in enumerate(route, start=1):
        plt.text(x, y, str(i), fontsize=8, ha="right", va="bottom")
    plt.title("TSP Route")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _parse_num_cities_spec(value: str) -> Tuple[int, int]:
    text = str(value).strip()
    for sep in ("-", ":", ","):
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) == 2:
                lo_val = int(float(parts[0]))
                hi_val = int(float(parts[1]))
                lo, hi = sorted((lo_val, hi_val))
                return (max(3, lo), max(3, hi))
    num = int(float(text))
    num = max(3, num)
    return (num, num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ppo_tsp.zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--max-decisions", type=str, default="200")
    parser.add_argument("--search-steps-per-decision", type=str, default="1")
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--tsp-num-cities", type=str, default="20")
    parser.add_argument("--tsp-grid-size", type=float, default=100.0)
    parser.add_argument("--tsp-seed", type=int, default=42)
    parser.add_argument("--tsp-coords-file", type=str, default=None)
    parser.add_argument("--tsp-distance-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs/tsp")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--route-image", type=str, default="route.png")
    parser.add_argument("--fitness-image", type=str, default="fitness.png")
    args = parser.parse_args()

    model = PPO.load(args.model_path, device='cpu')

    coords_arr = _load_array(args.tsp_coords_file)
    dist_arr = _load_array(args.tsp_distance_file)
    num_cities_range = _parse_num_cities_spec(args.tsp_num_cities)

    problem = TSPAdapter(
        num_cities=num_cities_range,
        grid_size=args.tsp_grid_size,
        seed=args.tsp_seed,
        coords=coords_arr.tolist() if coords_arr is not None else None,
        distance_matrix=dist_arr.tolist() if dist_arr is not None else None,
    )

    exploration = TSPMapElites(
        problem,
        population_size=32,
        bins_per_dim=(16, 16),
        random_injection_rate=0.15,
        seed=args.tsp_seed,
    )
    exploitation = TSPParticleSwarm(
        problem,
        population_size=32,
        omega=0.7,
        c1=1.5,
        c2=1.5,
        vmax=0.5,
        seed=args.tsp_seed,
    )
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

    episodes_info: list[dict] = []
    returns: list[float] = []

    logger = setup_logging('eval', 'tsp', log_dir=args.log_dir if args.log_dir else (Path(args.output_dir) / "logs"))

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

        # Per-episode meta snapshot
        coords_snapshot = np.asarray(env.orchestrator.problem.tsp_problem.city_coords, dtype=float).copy()
        logger.info(f"Episode {episode_idx} started. Num cities: {int(coords_snapshot.shape[0])}")

        while not done:
            phase_before = env.orchestrator.get_phase()
            action, _ = model.predict(obs, deterministic=args.deterministic)
            if int(action) == 1 and phase_before == "exploration":
                episode_switch_steps.append(step_idx)
            prev_best = env.orchestrator.get_best_solution()
            prev_best_fit = prev_best.fitness if prev_best else None
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            ep_return += reward
            candidate = env.orchestrator.get_best_solution()
            phase_after = env.orchestrator.get_phase()
            improvement = None
            if prev_best_fit is not None and candidate and candidate.fitness is not None:
                improvement = float(prev_best_fit - candidate.fitness)
            # Log step
            logger.info(f"Step {step_idx}: Phase before: {phase_before}, Action: {int(action)}, Phase after: {phase_after}, Reward: {float(reward):.3f}, Terminated: {bool(terminated)}, Truncated: {bool(truncated)}, Best fitness: {(float(candidate.fitness) if candidate and candidate.fitness is not None else None):.3f}, Improvement: {(float(improvement) if improvement is not None else None):.3f}")
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
        episodes_info.append({
            "index": episode_idx,
            "solution": episode_best_solution,
            "fitness": episode_best_fitness,
            "coords": coords_snapshot,
            "steps": episode_steps.copy(),
            "history": episode_fitness.copy(),
            "switch_steps": episode_switch_steps.copy(),
        })
        logger.info(f"Episode {episode_idx} ended. Total steps: {int(step_idx)}, Total return: {float(ep_return):.3f}, Best fitness: {float(episode_best_fitness):.3f}")

    env.close()

    if not episodes_info:
        print("No solution discovered during evaluation.")
        return

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if len(returns) > 1 else 0.0
    print(f"Evaluated {len(returns)} episode(s) | mean return: {mean_return:.3f} ± {std_return:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_base = Path(args.route_image)
    route_ext = route_base.suffix or ".png"
    route_stem = route_base.stem if route_base.suffix else route_base.name
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
        route_path = output_dir / f"{route_stem}_ep{idx}{route_ext}"
        _plot_tour(info["coords"], np.asarray(solution.representation, dtype=int), route_path)

        steps = info["steps"]
        history = info["history"]
        if steps and history:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, history, marker="o", linewidth=1.5)
            for switch_idx, step in enumerate(info["switch_steps"]):
                ax.axvline(step, color="red", linestyle="--", alpha=0.6, label="Phase switch" if switch_idx == 0 else "")
            if info["switch_steps"]:
                ax.legend()
            ax.set_title(f"Fitness over Decision Steps (episode {idx})")
            ax.set_xlabel("Decision step")
            ax.set_ylabel("Best fitness (lower is better)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(left=0, right=max(steps[-1] + 2, len(steps)))
            fig.tight_layout()
            fitness_path = output_dir / f"{fitness_stem}_ep{idx}{fitness_ext}"
            fig.savefig(fitness_path, dpi=150)
            plt.close(fig)

    print(f"Saved episode plots to {output_dir}")
    print(f"Step-by-step evaluation log: {logger.path()}")


if __name__ == "__main__":
    main()
