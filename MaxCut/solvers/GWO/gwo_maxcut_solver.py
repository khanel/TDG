"""
GWO-based Max-Cut Solver.

This module adapts the Gray Wolf Optimization (GWO) framework to solve the
unweighted Max-Cut problem defined in :mod:`MaxCut.maxcut`. It mirrors the TSP
solver structure while swapping the domain-specific operators for binary cut
partitions.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Core.problem import Solution
from GWO.GWO import GrayWolfOptimization
from MaxCut.maxcut import MaxCutProblem


@dataclass
class MaxCutGWOSolverConfig:
    """Configuration parameters for the MaxCutGWOSolver."""

    population_size: int = 50
    max_iterations: int = 200
    ensure_connected: bool = True
    mutation_rate_max: float = 0.35
    mutation_rate_min: float = 0.05
    visualize: bool = False
    plot_update_interval: int = 10
    pause_on_finish: float = 2.0


class MaxCutGWOSolver(GrayWolfOptimization):
    """
    Gray Wolf Optimization tailored to the binary Max-Cut problem.

    The solver keeps the overall structure of the TSP variant but works with
    binary masks representing graph partitions. Leaders drive exploitation by
    majority voting on bit assignments, while exploration introduces bit flips.
    """

    def __init__(
        self,
        weight_matrix: Iterable[Iterable[float]],
        *,
        config: Optional[MaxCutGWOSolverConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        cfg = config or MaxCutGWOSolverConfig()
        problem = MaxCutProblem(weight_matrix, seed=seed, ensure_connected=cfg.ensure_connected)
        super().__init__(problem, population_size=cfg.population_size, max_iterations=cfg.max_iterations)
        self._rng = np.random.default_rng(seed)
        self._config = cfg
        self._node_positions = None
        total_weight = float(np.sum(self.problem.weights))
        self._edge_weight_total = total_weight / 2.0
        self._random_baseline = self._edge_weight_total / 2.0 if self._edge_weight_total > 0 else 0.0
        self._live_fig = None
        self._live_ax = None
        self._visual_ready = False

    # --------------------------------------------------------------------- #
    # Core optimization loop
    # --------------------------------------------------------------------- #
    def step(self) -> None:
        # Guarantee evaluated population
        for sol in self.population:
            sol.evaluate()
        self.population.sort(key=lambda s: s.fitness if s.fitness is not None else float("inf"))
        if len(self.population) < 3:
            raise RuntimeError("Population size must be at least 3 for GWO leadership hierarchy.")

        alpha, beta, delta = (self.population[0], self.population[1], self.population[2])
        exploitation = self._exploitation_rate(self.iteration)

        new_population: List[Solution] = []
        for wolf in self.population:
            new_mask = self._update_binary_mask(
                wolf.representation,
                alpha.representation,
                beta.representation,
                delta.representation,
                exploitation,
            )
            child = Solution(new_mask, self.problem)
            child.evaluate()
            new_population.append(child)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _exploitation_rate(self, iteration: int) -> float:
        a = 2.0 - 2.0 * (iteration / max(1, self.max_iterations))
        return float(np.clip(1.0 - a / 2.0, 0.05, 1.0))

    def _mutation_rate(self, exploitation_rate: float) -> float:
        cfg = self._config
        # High exploitation -> low mutation; early iterations mutate more.
        rate = cfg.mutation_rate_min + (1.0 - exploitation_rate) * (cfg.mutation_rate_max - cfg.mutation_rate_min)
        return float(np.clip(rate, cfg.mutation_rate_min, cfg.mutation_rate_max))

    def _update_binary_mask(
        self,
        wolf: Iterable[int],
        alpha: Iterable[int],
        beta: Iterable[int],
        delta: Iterable[int],
        exploitation_rate: float,
    ) -> List[int]:
        wolf_vec = np.asarray(list(wolf), dtype=int)
        leaders = np.vstack(
            [np.asarray(list(alpha), dtype=int), np.asarray(list(beta), dtype=int), np.asarray(list(delta), dtype=int)]
        )
        leader_prob = np.mean(leaders, axis=0)

        n = wolf_vec.size
        inherit_mask = self._rng.random(n) < exploitation_rate
        new_bits = wolf_vec.copy()
        if inherit_mask.any():
            sampled = self._rng.random(inherit_mask.sum())
            new_bits[inherit_mask] = (sampled < leader_prob[inherit_mask]).astype(int)

        mutation_rate = self._mutation_rate(exploitation_rate)
        mutate_mask = self._rng.random(n) < mutation_rate
        new_bits[mutate_mask] = 1 - new_bits[mutate_mask]

        repaired = self.problem.repair(new_bits)
        return repaired.astype(int).tolist()

    # --------------------------------------------------------------------- #
    # Convenience running method
    # --------------------------------------------------------------------- #
    def optimize(
        self,
        *,
        max_iterations: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[Optional[Solution], List[float]]:
        """
        Run the optimization loop and return the best solution and fitness trace.
        """
        total_iters = max_iterations if max_iterations is not None else self.max_iterations
        if not self.population:
            self.initialize()

        history: List[float] = []
        for it in range(total_iters):
            self.step()
            best = self.get_best_solution()
            best_fit = float(best.fitness) if best and best.fitness is not None else float("inf")
            history.append(best_fit)
            if verbose and ((it + 1) % 10 == 0 or it == 0 or it + 1 == total_iters):
                cut_val = -best_fit if np.isfinite(best_fit) else float("-inf")
                quality = self._quality_metrics(cut_val)
                frac = quality["cut_fraction"]
                ratio = quality["random_ratio"]
                frac_str = f"{frac:.3f}" if math.isfinite(frac) else "n/a"
                ratio_str = f"{ratio:.2f}x" if math.isfinite(ratio) else "n/a"
                print(
                    f"[GWO MaxCut] Iter {it + 1}/{total_iters} | best cut={cut_val:.4f} "
                    f"| cut/m={frac_str} | vs random={ratio_str}"
                )
            if self._config.visualize and best is not None:
                if (it + 1) % max(1, self._config.plot_update_interval) == 0 or it == 0 or it + 1 == total_iters:
                    self._update_live_plot(best.representation, it + 1, -best_fit)

        if self._config.visualize and history:
            self._finalize_plot()

        return self.get_best_solution(), history

    def _quality_metrics(self, cut_weight: float) -> dict:
        metrics = {
            "cut_weight": cut_weight,
            "cut_fraction": float("nan"),
            "random_ratio": float("nan"),
        }
        if self._edge_weight_total > 0:
            metrics["cut_fraction"] = cut_weight / self._edge_weight_total
        if self._random_baseline > 0:
            metrics["random_ratio"] = cut_weight / self._random_baseline
        return metrics

    # ------------------------------------------------------------------ #
    # Visualization helpers
    # ------------------------------------------------------------------ #
    def _partition_layout(self, mask_arr: np.ndarray) -> np.ndarray:
        n = mask_arr.size
        positions = np.zeros((n, 2), dtype=float)

        clusters = {
            0: (-2.0, 0.0),
            1: (2.0, 0.0),
        }

        for part, center in clusters.items():
            indices = np.where(mask_arr == part)[0]
            if indices.size == 0:
                continue
            angles = np.linspace(0, 2 * np.pi, indices.size, endpoint=False)
            radius = 1.0 + 0.03 * max(0, indices.size - 1)
            cx, cy = center
            positions[indices, 0] = cx + radius * np.cos(angles)
            positions[indices, 1] = cy + radius * np.sin(angles)

        return positions

    def _ensure_plot(self) -> None:
        if self._visual_ready:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for visualization but is not installed.") from exc

        plt.ion()
        self._live_fig, self._live_ax = plt.subplots(figsize=(6, 6))
        self._live_ax.set_aspect("equal")
        self._live_ax.axis("off")
        self._live_fig.canvas.manager.set_window_title("GWO Max-Cut Progress")
        self._live_fig.show()
        self._visual_ready = True

    def _update_live_plot(self, mask: Iterable[int], iteration: int, best_cut: float) -> None:
        self._ensure_plot()
        import matplotlib.pyplot as plt

        mask_arr = np.asarray(list(mask), dtype=int)
        positions = self._partition_layout(mask_arr)
        ax = self._live_ax
        ax.clear()
        ax.set_title(f"GWO Max-Cut\nIteration {iteration} | Best cut {best_cut:.2f}")
        ax.axis("off")

        for i in range(positions.shape[0]):
            for j in range(i + 1, positions.shape[0]):
                if self.problem.weights[i, j] <= 0.0:
                    continue
                x_vals = [positions[i, 0], positions[j, 0]]
                y_vals = [positions[i, 1], positions[j, 1]]
                crossing = mask_arr[i] != mask_arr[j]
                color = "tab:red" if crossing else "lightgray"
                alpha = 0.9 if crossing else 0.45
                lw = 2.8 if crossing else 2.2
                ax.plot(x_vals, y_vals, color=color, alpha=alpha, linewidth=lw, zorder=1)

        colors = np.where(mask_arr > 0, "tab:blue", "tab:orange")
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=220, edgecolors="k", linewidths=1.2, zorder=2)
        for idx, (x, y) in enumerate(positions):
            ax.text(x, y, str(idx), ha="center", va="center", color="white", fontsize=11, zorder=3)

        self._live_fig.canvas.draw_idle()
        self._live_fig.canvas.flush_events()
        plt.pause(0.01)

    def _finalize_plot(self) -> None:
        if not self._visual_ready:
            return
        import matplotlib.pyplot as plt

        pause = max(0.0, float(self._config.pause_on_finish))
        if pause > 0:
            plt.pause(pause)
        plt.ioff()
        plt.show(block=False)


__all__ = ["MaxCutGWOSolver", "MaxCutGWOSolverConfig"]


def _load_adjacency(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"matrix file not found: {path}")
    if path.suffix in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data[next(iter(data.files))]
    else:
        try:
            data = np.loadtxt(path, delimiter=",")
        except ValueError:
            data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if not np.allclose(data, data.T, atol=1e-8):
        raise ValueError("adjacency matrix must be symmetric")
    data = np.where(data > 0.5, 1.0, 0.0)
    np.fill_diagonal(data, 0.0)
    return data


def _generate_random_matrix(n_nodes: int, edge_prob: float, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n_nodes, n_nodes)) < edge_prob
    upper = np.triu(mask, k=1).astype(float)
    mat = upper + upper.T
    np.fill_diagonal(mat, 0.0)
    return mat


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Gray Wolf Optimization for the Max-Cut problem.")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "-m",
        "--matrix",
        type=str,
        help="Path to a symmetric 0/1 adjacency matrix (.npy, .npz, .csv, or whitespace-separated).",
    )
    src_group.add_argument(
        "-n",
        "--random-nodes",
        type=int,
        help="Generate a random graph with the given number of nodes.",
    )
    parser.add_argument(
        "-p",
        "--edge-prob",
        type=float,
        default=0.3,
        help="Edge probability for random graph generation (default: 0.3).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="GWO population size (default: 50).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Maximum number of iterations (default: 200).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable live visualization of the cut evolution.",
    )
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=10,
        help="Iteration interval for plot updates when visualization is enabled.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Seconds to pause on the final visualization frame (default: 2.0).",
    )
    parser.add_argument(
        "--mutation-max",
        type=float,
        default=0.35,
        help="Maximum mutation rate during exploration (default: 0.35).",
    )
    parser.add_argument(
        "--mutation-min",
        type=float,
        default=0.05,
        help="Minimum mutation rate during exploitation (default: 0.05).",
    )

    args = parser.parse_args(argv)

    if args.matrix:
        adjacency = _load_adjacency(Path(args.matrix))
    else:
        if args.random_nodes is None or args.random_nodes <= 1:
            raise ValueError("random graph generation requires --random-nodes >= 2")
        adjacency = _generate_random_matrix(args.random_nodes, args.edge_prob, args.seed)

    cfg = MaxCutGWOSolverConfig(
        population_size=max(3, int(args.population)),
        max_iterations=max(1, int(args.iterations)),
        ensure_connected=True,
        mutation_rate_max=float(args.mutation_max),
        mutation_rate_min=float(args.mutation_min),
        visualize=bool(args.visualize),
        plot_update_interval=max(1, int(args.plot_interval)),
        pause_on_finish=float(args.pause),
    )

    solver = MaxCutGWOSolver(adjacency, config=cfg, seed=args.seed)
    best, history = solver.optimize(verbose=True)

    if best is None or best.fitness is None:
        print("\nNo feasible solution found.")
    else:
        cut_weight = -best.fitness
        mask_str = " ".join(str(int(v)) for v in best.representation)
        quality = solver._quality_metrics(cut_weight)
        frac = quality["cut_fraction"]
        ratio = quality["random_ratio"]
        frac_str = f"{frac:.3f}" if math.isfinite(frac) else "n/a"
        ratio_str = f"{ratio:.2f}x" if math.isfinite(ratio) else "n/a"
        print("\nBest cut weight:", cut_weight)
        print("Partition mask:", mask_str)
        print("Iterations run:", len(history))
        print(f"Total edges: {solver._edge_weight_total:.0f} | Expected random cut: {solver._random_baseline:.2f}")
        print(f"Cut density (cut/m): {frac_str} | Gain vs random baseline: {ratio_str}")


if __name__ == "__main__":
    main()
