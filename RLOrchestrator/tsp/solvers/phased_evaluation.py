"""
Evaluation script for a simple phased search strategy (50% exploration, 50% exploitation).
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ...core.orchestrator import Orchestrator
from ...tsp.adapter import TSPAdapter
from ...tsp.solvers import TSPMapElites, TSPSimulatedAnnealing


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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-search-steps", type=int, default=200)
    parser.add_argument("--tsp-num-cities", type=str, default="20")
    parser.add_argument("--tsp-grid-size", type=float, default=100.0)
    parser.add_argument("--tsp-seed", type=int, default=42)
    parser.add_argument("--tsp-coords-file", type=str, default=None)
    parser.add_argument("--tsp-distance-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs/tsp_phased")
    parser.add_argument("--route-image", type=str, default="route.png")
    parser.add_argument("--fitness-image", type=str, default="fitness.png")
    args = parser.parse_args()

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

    exploration_steps = args.max_search_steps // 2
    exploitation_steps = max(1, args.max_search_steps - exploration_steps)
    sa_moves_per_temp = 50
    sa_max_iterations = max(sa_moves_per_temp * exploitation_steps, sa_moves_per_temp)

    episodes_info: list[dict] = []

    for episode_idx in range(1, max(1, args.episodes) + 1):
        episode_seed = args.tsp_seed + episode_idx

        problem = TSPAdapter(
            num_cities=num_cities_range,
            grid_size=args.tsp_grid_size,
            seed=episode_seed,
            coords=coords_arr.tolist() if coords_arr is not None else None,
            distance_matrix=dist_arr.tolist() if dist_arr is not None else None,
        )

        exploration = TSPMapElites(
            problem,
            population_size=32,
            bins_per_dim=(16, 16),
            random_injection_rate=0.15,
            seed=episode_seed,
        )
        exploitation = TSPSimulatedAnnealing(
            problem,
            population_size=1,
            initial_temperature=100.0,  # Higher temperature for more exploration
            final_temperature=1e-3,
            cooling_rate=0.99,  # Slower cooling for gradual exploitation
            moves_per_temp=sa_moves_per_temp,  # More moves per temperature level
            max_iterations=sa_max_iterations,
            seed=episode_seed,
        )
        for solver in (exploration, exploitation):
            if hasattr(solver, "initialize"):
                solver.initialize()

        orchestrator = Orchestrator(problem, exploration, exploitation, start_phase="exploration")
        orchestrator._update_best()

        episode_steps: list[int] = []
        episode_fitness: list[float] = []
        episode_best_solution = None
        episode_best_fitness = float("inf")

        # Exploration phase
        for step_idx in range(exploration_steps):
            orchestrator.step()
            candidate = orchestrator.get_best_solution()
            if candidate and candidate.fitness is not None:
                episode_steps.append(step_idx)
                episode_fitness.append(candidate.fitness)
                if candidate.fitness < episode_best_fitness:
                    episode_best_solution = candidate.copy()
                    episode_best_fitness = candidate.fitness

        # Switch to exploitation, seeding with the best solution from exploration
        best_exploration_solution = orchestrator.get_best_solution()
        if best_exploration_solution:
            orchestrator.switch_to_exploitation(seeds=[best_exploration_solution])
        else:
            orchestrator.switch_to_exploitation()  # Fallback if no solution found

        # Exploitation phase
        for step_idx in range(exploration_steps, args.max_search_steps):
            orchestrator.step()
            candidate = orchestrator.get_best_solution()
            if candidate and candidate.fitness is not None:
                episode_steps.append(step_idx)
                episode_fitness.append(candidate.fitness)
                if candidate.fitness < episode_best_fitness:
                    episode_best_solution = candidate.copy()
                    episode_best_fitness = candidate.fitness
        
        if episode_best_solution is None:
            episode_best_solution = orchestrator.get_best_solution().copy()
            episode_best_fitness = episode_best_solution.fitness if episode_best_solution else float("inf")

        coords_snapshot = np.asarray(orchestrator.problem.tsp_problem.city_coords, dtype=float).copy()
        episodes_info.append({
            "index": episode_idx,
            "solution": episode_best_solution,
            "fitness": episode_best_fitness,
            "coords": coords_snapshot,
            "steps": episode_steps.copy(),
            "history": episode_fitness.copy(),
            "switch_steps": [exploration_steps],
        })

    if not episodes_info:
        print("No solution discovered during evaluation.")
        return

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


if __name__ == "__main__":
    main()
