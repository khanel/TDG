"""
Lightweight evaluation for the composite-reward TSP policy.

Runs a trained model across multiple random TSP sizes and saves route plots
under temp/composite_eval/.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

# Ensure project root is importable (shared with training script)
import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use the updated training environment builder and the new reward config
from composite_reward_training import build_env, _parse_int_or_range  # noqa: E402
from RLOrchestrator.core.reward import RewardConfig  # noqa: E402


def _parse_sizes(text: str) -> List[int]:
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return [40]
    return [max(3, int(float(p))) for p in parts]


def _plot_route(coords: np.ndarray, tour: Iterable[int], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tour_idx = np.asarray(tour, dtype=int) - 1
    route = coords[tour_idx]
    closed = np.vstack([route, route[0]])
    plt.figure(figsize=(6, 6))
    plt.plot(closed[:, 0], closed[:, 1], "-o", markersize=4)
    for idx, (x, y) in enumerate(route, start=1):
        plt.text(x, y, str(idx), fontsize=8, ha="right", va="bottom")
    plt.title("TSP Route")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate(
    model_path: str,
    num_cities_list: List[int],
    episodes_per_size: int,
    output_dir: Path,
    *,
    problem_name: str = "tsp",
    grid_size: float = 120.0,
    max_decision_steps: int | tuple[int, int] = 50,
    search_steps_per_decision: int | tuple[int, int] = 1,
    max_search_steps: int | None = None,
    device: str = "cpu",
) -> None:
    model = PPO.load(model_path, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.txt"
    summary_file.write_text("# Evaluation Summary (updating)\n")

    episode_logs: list[str] = []

    # Per-size aggregations (fitness only compared within same city count)
    per_size_stats: dict[int, dict[str, dict]] = defaultdict(
        lambda: {
            "pairs": defaultdict(lambda: {"episodes": 0, "reward_sum": 0.0, "fitness_sum": 0.0, "best_fitness_min": float("inf")}),
            "explorers": defaultdict(lambda: {"episodes": 0, "reward_sum": 0.0, "fitness_sum": 0.0, "best_fitness_min": float("inf")}),
            "exploiters": defaultdict(lambda: {"episodes": 0, "reward_sum": 0.0, "fitness_sum": 0.0, "best_fitness_min": float("inf")}),
        }
    )
    # Cross-size reward uses reward normalized by city count to reflect rising difficulty
    overall_explorer_reward_per_city: dict[str, dict] = defaultdict(lambda: {"episodes": 0, "reward_per_city_sum": 0.0})
    overall_exploiter_reward_per_city: dict[str, dict] = defaultdict(lambda: {"episodes": 0, "reward_per_city_sum": 0.0})
    overall_pair_reward_per_city: dict[Tuple[str, str], dict] = defaultdict(lambda: {"episodes": 0, "reward_per_city_sum": 0.0})

    def _select_best_reward(stats_map):
        if not stats_map:
            return None
        return max(
            stats_map.items(),
            key=lambda item: (item[1]["reward_sum"] / max(1, item[1]["episodes"])),
        )

    def _select_best_fitness(stats_map):
        candidates = []
        for name, vals in stats_map.items():
            if vals["episodes"] > 0 and vals["fitness_sum"] > 0.0:
                avg_fit = vals["fitness_sum"] / vals["episodes"]
            else:
                avg_fit = vals["best_fitness_min"]
            candidates.append((name, avg_fit, vals))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])

    def _select_best_reward_per_city(stats_map):
        if not stats_map:
            return None
        return max(
            stats_map.items(),
            key=lambda item: (item[1]["reward_per_city_sum"] / max(1, item[1]["episodes"])),
        )

    def _write_summary():
        summary_lines = [
            "# Aggregated Solver Summary",
            "# Per-city results (reward: higher better; fitness only compared within the same city count)",
        ]
        for n_cities in sorted(per_size_stats.keys()):
            size_stats = per_size_stats[n_cities]
            best_pair_reward = _select_best_reward(size_stats["pairs"])
            best_pair_fitness = _select_best_fitness(size_stats["pairs"])
            best_explorer = _select_best_reward(size_stats["explorers"])
            best_explorer_fit = _select_best_fitness(size_stats["explorers"])
            best_exploiter = _select_best_reward(size_stats["exploiters"])
            best_exploiter_fit = _select_best_fitness(size_stats["exploiters"])

            summary_lines.append(f"- cities={n_cities}:")
            if best_pair_reward:
                pair_name, vals = best_pair_reward
                summary_lines.append(
                    f"    pair (reward): {pair_name[0]} + {pair_name[1]} | avg_reward={vals['reward_sum']/max(1,vals['episodes']):.3f} | episodes={vals['episodes']}"
                )
            if best_pair_fitness:
                name, avg_fit, vals = best_pair_fitness
                summary_lines.append(
                    f"    pair (fitness): {name[0]} + {name[1]} | avg_best_fitness={avg_fit:.4f} | episodes={vals['episodes']}"
                )
            if best_explorer:
                name, vals = best_explorer
                summary_lines.append(
                    f"    explorer (reward): {name} | avg_reward={vals['reward_sum']/max(1,vals['episodes']):.3f} | episodes={vals['episodes']}"
                )
            if best_explorer_fit:
                name, avg_fit, vals = best_explorer_fit
                summary_lines.append(
                    f"    explorer (fitness): {name} | avg_best_fitness={avg_fit:.4f} | episodes={vals['episodes']}"
                )
            if best_exploiter:
                name, vals = best_exploiter
                summary_lines.append(
                    f"    exploiter (reward): {name} | avg_reward={vals['reward_sum']/max(1,vals['episodes']):.3f} | episodes={vals['episodes']}"
                )
            if best_exploiter_fit:
                name, avg_fit, vals = best_exploiter_fit
                summary_lines.append(
                    f"    exploiter (fitness): {name} | avg_best_fitness={avg_fit:.4f} | episodes={vals['episodes']}"
                )

        summary_lines.append(
            "# Cross-size (reward normalized by city count; fitness not compared across city counts)"
        )
        best_pair_overall = _select_best_reward_per_city(overall_pair_reward_per_city)
        best_explorer_overall = _select_best_reward_per_city(overall_explorer_reward_per_city)
        best_exploiter_overall = _select_best_reward_per_city(overall_exploiter_reward_per_city)

        if best_pair_overall:
            name, vals = best_pair_overall
            summary_lines.append(
                f"- overall pair: {name[0]} + {name[1]} | avg_reward_per_city={vals['reward_per_city_sum']/max(1,vals['episodes']):.4f} | episodes={vals['episodes']}"
            )
        if best_explorer_overall:
            name, vals = best_explorer_overall
            summary_lines.append(
                f"- overall explorer: {name} | avg_reward_per_city={vals['reward_per_city_sum']/max(1,vals['episodes']):.4f} | episodes={vals['episodes']}"
            )
        if best_exploiter_overall:
            name, vals = best_exploiter_overall
            summary_lines.append(
                f"- overall exploiter: {name} | avg_reward_per_city={vals['reward_per_city_sum']/max(1,vals['episodes']):.4f} | episodes={vals['episodes']}"
            )

        summary_lines.append("# Episode details")

        with summary_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
            for block in episode_logs:
                f.write(block)

    for n_cities in num_cities_list:
        for ep in range(episodes_per_size):
            # Use the elastic reward defaults
            reward_cfg = RewardConfig()
            adapter_kwargs = None
            if problem_name.lower() == "tsp":
                adapter_kwargs = {"num_cities": n_cities, "grid_size": grid_size}

            env = build_env(
                problem_name=problem_name,
                adapter_kwargs=adapter_kwargs,
                max_decision_steps=max_decision_steps,
                search_steps_per_decision=search_steps_per_decision,
                max_search_steps=max_search_steps,
                reward_config=reward_cfg,
            )
            obs, _ = env.reset()

            # --- Extract detailed config for logging ---
            explorer = env.exploration_solver
            exploiter = env.exploitation_solver
            explorer_name = explorer.__class__.__name__
            exploiter_name = exploiter.__class__.__name__
            explorer_pop = getattr(explorer, 'population_size', 'N/A')
            exploiter_pop = getattr(exploiter, 'population_size', 'N/A')
            ep_max_decisions = env._context.max_decision_steps
            ep_search_steps = env._context.search_steps_per_decision
            # ---

            done = False
            total_reward = 0.0
            steps = 0
            total_evals = 0
            action_counts = {
                "exploration": {"stay": 0, "advance": 0},
                "exploitation": {"stay": 0, "advance": 0},
            }

            while not done:
                phase_before = getattr(env, "get_phase", lambda: "unknown")()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(int(action))
                done = bool(terminated or truncated)
                total_reward += float(reward)
                total_evals = _info.get("total_evals", total_evals)
                steps += 1

                phase_key = str(phase_before)
                if phase_key not in action_counts:
                    action_counts[phase_key] = {"stay": 0, "advance": 0}
                if int(action) == 0:
                    action_counts[phase_key]["stay"] += 1
                else:
                    action_counts[phase_key]["advance"] += 1

            best = env.get_best_solution()
            coords = np.asarray(env.problem.tsp_problem.city_coords, dtype=float)
            tour = best.representation if best is not None else list(range(1, len(coords) + 1))

            tag = f"{n_cities}c_ep{ep+1}"
            route_path = output_dir / f"route_{tag}.png"
            _plot_route(coords, tour, route_path)

            summary_block = f"""
tag: {tag}
steps: {steps}
reward: {total_reward:.3f}
best_fitness: {getattr(best, 'fitness', 'N/A')}
total_evaluations: {total_evals}
config_max_decisions: {ep_max_decisions}
config_search_steps_per_decision: {ep_search_steps}
explorer: {explorer_name}(pop={explorer_pop})
exploiter: {exploiter_name}(pop={exploiter_pop})
actions_explore: stay={action_counts['exploration']['stay']},adv={action_counts['exploration']['advance']}
actions_exploit: stay={action_counts['exploitation']['stay']},adv={action_counts['exploitation']['advance']}
"""
            episode_logs.append(summary_block)
            env.close()

            # Update per-size stats
            size_stats = per_size_stats[n_cities]
            pair_key = (explorer_name, exploiter_name)
            for group_key, key in (("pairs", pair_key), ("explorers", explorer_name), ("exploiters", exploiter_name)):
                bucket = size_stats[group_key][key]
                bucket["episodes"] += 1
                bucket["reward_sum"] += total_reward
                if best is not None and getattr(best, "fitness", None) is not None:
                    fit_val = float(best.fitness)
                    bucket["fitness_sum"] += fit_val
                    bucket["best_fitness_min"] = min(bucket["best_fitness_min"], fit_val)

            # Cross-size reward normalized by city count (higher reward per city is better)
            reward_per_city = total_reward / float(max(1, n_cities))
            overall_explorer_reward_per_city[explorer_name]["episodes"] += 1
            overall_explorer_reward_per_city[explorer_name]["reward_per_city_sum"] += reward_per_city
            overall_exploiter_reward_per_city[exploiter_name]["episodes"] += 1
            overall_exploiter_reward_per_city[exploiter_name]["reward_per_city_sum"] += reward_per_city
            overall_pair_reward_per_city[pair_key]["episodes"] += 1
            overall_pair_reward_per_city[pair_key]["reward_per_city_sum"] += reward_per_city

            # Persist incremental summary after each episode
            _write_summary()

    # Final write in case no episodes ran (e.g., empty inputs)
    _write_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="temp/ppo_elastic_reward.zip")
    parser.add_argument("--problem-name", type=str, default="tsp", help="Problem to evaluate (default: tsp)")
    parser.add_argument("--num-cities", type=str, default="30,60,90")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per city count")
    parser.add_argument("--output-dir", type=str, default="temp/composite_eval")
    parser.add_argument("--grid-size", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-decision-steps", type=str, default="40-160")
    parser.add_argument("--search-steps-per-decision", type=str, default="1-3")
    parser.add_argument("--max-search-steps", type=int, default=None)
    args = parser.parse_args()

    max_decisions = _parse_int_or_range(args.max_decision_steps, minimum=10)
    search_steps = _parse_int_or_range(args.search_steps_per_decision, minimum=1)

    evaluate(
        model_path=args.model_path,
        num_cities_list=_parse_sizes(args.num_cities),
        episodes_per_size=max(1, args.episodes),
        output_dir=Path(args.output_dir),
        problem_name=args.problem_name,
        grid_size=args.grid_size,
        max_decision_steps=max_decisions,
        search_steps_per_decision=search_steps,
        max_search_steps=args.max_search_steps,
        device=args.device,
    )
