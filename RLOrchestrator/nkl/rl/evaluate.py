"""
Evaluation script for trained NKL orchestrator policies.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ...core.orchestrator import Orchestrator
from ...core.utils import parse_int_range, setup_logging
from ...core.env_factory import create_env
from ..adapter import NKLAdapter
from ..solvers.explorer import NKLRandomExplorer
from ..solvers.local_search import NKLLocalSearch


def _plot_fitness_history(steps: list[int], history: list[float], switch_steps: list[int], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history, marker="o", linewidth=1.5)
    for switch_idx, step in enumerate(switch_steps):
        ax.axvline(step, color="red", linestyle="--", alpha=0.6, label="Phase switch" if switch_idx == 0 else "")
    if switch_steps:
        ax.legend()
    ax.set_title("Fitness over Decision Steps")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if steps:
        ax.set_xlim(left=0, right=max(steps[-1] + 2, len(steps)))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ppo_nkl.zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--max-decisions", type=str, default="200")
    parser.add_argument("--search-steps-per-decision", type=str, default="10")
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--nkl-n-items", type=str, default="100")
    parser.add_argument("--nkl-k-interactions", type=str, default="5")
    parser.add_argument("--nkl-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs/nkl")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--fitness-image", type=str, default="fitness.png")
    args = parser.parse_args()

    session_id = int(time.time())
    logger = setup_logging('eval', 'nkl', log_dir=(args.log_dir or 'logs'), session_id=session_id)

    model = PPO.load(args.model_path, device='cpu')

    n_items_range = parse_int_range(args.nkl_n_items, min_value=2, label="nkl-n-items")
    k_interactions_range = parse_int_range(args.nkl_k_interactions, min_value=0, label="nkl-k-interactions")

    problem = NKLAdapter(
        n_items=n_items_range,
        k_interactions=k_interactions_range,
        seed=args.nkl_seed,
    )

    exploration = NKLRandomExplorer(
        problem,
        population_size=64,
        flip_probability=0.15,
        elite_fraction=0.25,
        seed=args.nkl_seed,
    )
    exploitation = NKLLocalSearch(
        problem,
        population_size=16,
        moves_per_step=8,
        escape_probability=0.05,
        seed=args.nkl_seed,
    )
    for solver in (exploration, exploitation):
        if hasattr(solver, "initialize"):
            solver.initialize()

    orchestrator = Orchestrator(problem, exploration, exploitation, start_phase="exploration")
    orchestrator._update_best()
    max_decision_spec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec = parse_int_range(args.search_steps_per_decision, min_value=1, label="search-steps-per-decision")
    env = create_env(
        problem,
        exploration,
        exploitation,
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
        f"Run: mode=eval, session_id={session_id}, problem=nkl, episodes={int(args.episodes)}"
    )
    logger.info(
        f"Config: max_decisions={args.max_decisions}, steps_per_decision={args.search_steps_per_decision}, reward_clip={args.reward_clip}"
    )

    episodes_info: list[dict] = []
    returns: list[float] = []

    for episode_idx in range(1, max(1, args.episodes) + 1):
        logger.debug(f"Episode {episode_idx} started. Problem info: {problem.get_problem_info()}")
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
            phase_before = env.orchestrator.get_phase()
            action, _ = model.predict(obs, deterministic=args.deterministic)
            if int(action) == 1 and env.orchestrator.get_phase() == "exploration":
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
            logger.debug(
                f"Step {step_idx}: phase_before={phase_before}, action={int(action)}, phase_after={phase_after}, reward={float(reward):.3f}, terminated={bool(terminated)}, truncated={bool(truncated)}, best={(float(candidate.fitness) if candidate and candidate.fitness is not None else None):.3f}, improvement={(float(improvement) if improvement is not None else None):.3f}"
            )
            step_idx += 1

        returns.append(ep_return)
        if episode_best_solution is None:
            episode_best_solution = env.orchestrator.get_best_solution().copy()
            episode_best_fitness = episode_best_solution.fitness if episode_best_solution else float("inf")

        episodes_info.append({
            "index": episode_idx,
            "solution": episode_best_solution,
            "fitness": episode_best_fitness,
            "steps": episode_steps.copy(),
            "history": episode_fitness.copy(),
            "switch_steps": episode_switch_steps.copy(),
        })
        logger.debug(f"Episode {episode_idx} ended. steps={int(step_idx)}, return={float(ep_return):.3f}, best={float(episode_best_fitness):.3f}")
    env.close()

    if not episodes_info:
        logger.info("Summary: episodes=0, return_mean=0.0000, return_std=0.0000, return_min=0.0000, return_max=0.0000, duration_sec=0.0")
        return

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if len(returns) > 1 else 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fitness_base = Path(args.fitness_image)
    fitness_ext = fitness_base.suffix or ".png"
    fitness_stem = fitness_base.stem if fitness_base.suffix else fitness_base.name

    best_episode = min(episodes_info, key=lambda d: d["fitness"] if d["fitness"] is not None else float("inf"))
    # No console prints

    for info in episodes_info:
        idx = info["index"]
        steps = info["steps"]
        history = info["history"]
        if steps and history:
            fitness_path = output_dir / f"{fitness_stem}_ep{idx}{fitness_ext}"
            _plot_fitness_history(steps, history, info["switch_steps"], fitness_path)

    # Final one-line summary for evaluation
    duration = max(0.0, time.time() - start_time)
    logger.info(
        f"Summary: episodes={len(returns)}, return_mean={mean_return:.4f}, return_std={std_return:.4f}, "
        f"return_min={float(np.min(returns)):.4f}, return_max={float(np.max(returns)):.4f}, duration_sec={duration:.1f}"
    )


if __name__ == "__main__":
    main()
