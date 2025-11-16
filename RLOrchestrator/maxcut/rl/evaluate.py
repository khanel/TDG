"""
Evaluation script for trained Max-Cut orchestrator policies.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ...core.utils import parse_int_range, setup_logging
from ...core.env_factory import create_env
from ...problems.registry import instantiate_problem
from ...rl.eval_logging import EvaluationLogger, StepRecord, EpisodeSummary


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


def _stage_map(stages):
    mapping = {binding.name: binding.solver for binding in stages}
    missing = {"exploration", "exploitation"} - mapping.keys()
    if missing:
        raise ValueError(f"Problem bundle missing stages: {sorted(missing)}")
    return mapping


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
    parser.add_argument("--max-decisions", type=str, default="200")
    parser.add_argument("--search-steps-per-decision", type=str, default="1")
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--n-nodes", type=int, default=64)
    parser.add_argument("--edge-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ensure-connected", action="store_true", default=False)
    parser.add_argument("--weights-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs/maxcut")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--partition-image", type=str, default="partition.png")
    parser.add_argument("--fitness-image", type=str, default="fitness.png")
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    weights = _load_weights(args.weights_file)
    adapter_kwargs = {
        "weight_matrix": weights.tolist() if isinstance(weights, np.ndarray) else weights,
        "n_nodes": args.n_nodes,
        "edge_probability": args.edge_probability,
        "seed": args.seed,
        "ensure_connected": args.ensure_connected,
    }
    solver_overrides = {
        "exploration": {
            "population_size": 64,
            "flip_probability": 0.15,
            "elite_fraction": 0.25,
            "seed": args.seed,
        },
        "exploitation": {
            "population_size": 16,
            "moves_per_step": 8,
            "escape_probability": 0.05,
            "seed": args.seed,
        },
    }
    bundle = instantiate_problem(
        "maxcut",
        adapter_kwargs=adapter_kwargs,
        solver_kwargs=solver_overrides,
    )
    stage_map = _stage_map(bundle.stages)
    for solver in stage_map.values():
        if hasattr(solver, "initialize"):
            solver.initialize()

    session_id = int(time.time())
    logger = setup_logging('eval', 'maxcut', log_dir=(args.log_dir or 'logs'), session_id=session_id)

    max_decision_spec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec = parse_int_range(args.search_steps_per_decision, min_value=1, label="search-steps-per-decision")
    env = create_env(
        bundle.problem,
        stage_map["exploration"],
        stage_map["exploitation"],
        max_decision_steps=max_decision_spec,
        search_steps_per_decision=search_step_spec,
        max_search_steps=args.max_search_steps,
        reward_clip=args.reward_clip,
        log_type='eval',
        log_dir=(args.log_dir or 'logs'),
        session_id=session_id,
        emit_init_summary=True,
    )

    # High-level start header and config (single lines)
    start_time = time.time()
    logger.info(
        f"Run: mode=eval, session_id={session_id}, problem=maxcut, episodes={int(args.episodes)}"
    )
    logger.info(
        f"Config: max_decisions={args.max_decisions}, steps_per_decision={args.search_steps_per_decision}, reward_clip={args.reward_clip}"
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

        # Episode meta snapshot
        n_nodes = int(env.problem.get_problem_info().get("dimension", 0))
        logger.info(f"Episode {episode_idx} started. Num nodes: {n_nodes}")

        while not done:
            phase_before = env.get_phase()
            action, _ = model.predict(obs, deterministic=args.deterministic)
            if int(action) == 1 and phase_before == "exploration":
                episode_switch_steps.append(step_idx)
            # Log special events at INFO level with observation snapshot
            if int(action) == 1:
                event_type = "switch" if phase_before == "exploration" else "terminate"
                try:
                    logger.info(
                        f"Event: action=1, type={event_type}, step={step_idx}, phase_before={phase_before}, observation={(obs.tolist() if hasattr(obs,'tolist') else list(obs))}"
                    )
                except Exception:
                    pass
            prev_best = env.get_best_solution()
            prev_best_fit = prev_best.fitness if prev_best else None
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            ep_return += reward
            candidate = env.get_best_solution()
            phase_after = env.get_phase()
            improvement = None
            if prev_best_fit is not None and candidate and candidate.fitness is not None:
                improvement = float(prev_best_fit - candidate.fitness)
            logger.debug(
                f"Step {step_idx}: phase_before={phase_before}, action={int(action)}, phase_after={phase_after}, reward={float(reward):.3f}, terminated={bool(terminated)}, truncated={bool(truncated)}, best={(float(candidate.fitness) if candidate and candidate.fitness is not None else None):.3f}, improvement={(float(improvement) if improvement is not None else None):.3f}"
            )
            if candidate and candidate.fitness is not None:
                episode_steps.append(step_idx)
                episode_fitness.append(candidate.fitness)
                if candidate.fitness < episode_best_fitness:
                    episode_best_solution = candidate.copy()
                    episode_best_fitness = candidate.fitness
            step_idx += 1
        returns.append(ep_return)

        if episode_best_solution is None:
            current_best = env.get_best_solution()
            episode_best_solution = current_best.copy() if current_best is not None else None
            episode_best_fitness = episode_best_solution.fitness if episode_best_solution else float("inf")
        weights_snapshot = np.asarray(env.problem.maxcut_problem.weights, dtype=float).copy()
        episodes_info.append({
            "index": episode_idx,
            "solution": episode_best_solution,
            "fitness": episode_best_fitness,
            "weights": weights_snapshot,
            "steps": episode_steps.copy(),
            "history": episode_fitness.copy(),
            "switch_steps": episode_switch_steps.copy(),
        })
        logger.info(f"Episode {episode_idx} ended. Total steps: {int(step_idx)}, Total return: {float(ep_return):.3f}, Best fitness: {float(episode_best_fitness):.3f}")

    env.close()

    if not episodes_info:
        logger.info("Summary: episodes=0, return_mean=0.0000, return_std=0.0000, return_min=0.0000, return_max=0.0000, duration_sec=0.0")
        return

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if len(returns) > 1 else 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_base = Path(args.partition_image)
    partition_ext = partition_base.suffix or ".png"
    partition_stem = partition_base.stem if partition_base.suffix else partition_base.name
    fitness_base = Path(args.fitness_image)
    fitness_ext = fitness_base.suffix or ".png"
    fitness_stem = fitness_base.stem if fitness_base.suffix else fitness_base.name

    best_episode = min(episodes_info, key=lambda d: d["fitness"] if d["fitness"] is not None else float("inf"))
    # No console prints

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
                # no console prints
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

    # Final one-line summary for evaluation
    duration = max(0.0, time.time() - start_time)
    logger.info(
        f"Summary: episodes={len(returns)}, return_mean={mean_return:.4f}, return_std={std_return:.4f}, "
        f"return_min={float(np.min(returns)):.4f}, return_max={float(np.max(returns)):.4f}, duration_sec={duration:.1f}"
    )


if __name__ == "__main__":
    main()
