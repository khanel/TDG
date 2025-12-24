"""Compare solver pairs by running many evaluation episodes.

Goal: run N episodes per (exploration_solver, exploitation_solver) pair and
rank pairs by mean best fitness (lower is better, consistent with Solution
ordering used by the orchestrator).

This is a baseline comparator: it does NOT use an RL policy. It uses a simple,
consistent control policy:
- Switch from exploration -> exploitation halfway through the decision budget
- Terminate (advance to termination) on the final decision

This isolates solver-pair performance under a fixed orchestration rule.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from RLOrchestrator.core.env_factory import create_env
from RLOrchestrator.core.utils import IntRangeSpec, parse_int_range
from RLOrchestrator.problems.registry import get_problem_definition


@dataclass(frozen=True)
class PairResult:
    explorer: str
    exploiter: str
    episodes: int
    fitness_mean: float
    fitness_std: float
    fitness_min: float
    fitness_max: float


def parse_num_cities_choices(value: str) -> List[int]:
    """Parse comma-separated fixed city counts (e.g., '20,50,200').

    Intended for TSP only. This deliberately does NOT accept ranges.
    """
    if value is None:
        return []

    raw = str(value)
    tokens = [t.strip() for t in raw.split(",")]
    out: List[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not token:
            continue
        n = int(token)
        if n < 3:
            raise ValueError(f"num-cities must be >= 3 (got {n})")
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    if not out:
        raise ValueError("num-cities list is empty")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare solver pairs by repeated evaluation.")
    parser.add_argument("--problem", type=str, default="tsp")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--num-cities",
        type=str,
        default=None,
        help="TSP only: comma-separated fixed city counts (e.g., '20,50,200').",
    )

    parser.add_argument("--max-decisions", type=str, default="200")
    parser.add_argument("--search-steps-per-decision", type=str, default="1")
    parser.add_argument("--max-search-steps", type=int, default=None)
    return parser


def list_solver_pairs(problem: str) -> List[Tuple[str, str]]:
    """Return all (explorer_cls_name, exploiter_cls_name) pairs for a problem."""
    definition = get_problem_definition(str(problem))
    if definition is None:
        raise KeyError(f"Unknown problem: {problem}")

    exp_spec = definition.solvers.get("exploration")
    imp_spec = definition.solvers.get("exploitation")

    explorers = _as_factories(exp_spec)
    exploiters = _as_factories(imp_spec)

    pairs: List[Tuple[str, str]] = []
    for e in explorers:
        for x in exploiters:
            pairs.append((e.cls.__name__, x.cls.__name__))
    return pairs


def main() -> None:
    args = build_parser().parse_args()

    definition = get_problem_definition(str(args.problem))
    if definition is None:
        raise SystemExit(f"Unknown problem: {args.problem}")

    episodes = max(1, int(args.episodes))
    base_seed = int(args.seed)

    num_cities_choices: Optional[List[int]] = None
    if args.num_cities is not None:
        if str(args.problem) != "tsp":
            raise SystemExit("--num-cities is only supported for --problem tsp")
        try:
            num_cities_choices = parse_num_cities_choices(args.num_cities)
        except ValueError as e:
            raise SystemExit(str(e))

    max_decision_spec: IntRangeSpec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec: IntRangeSpec = parse_int_range(
        args.search_steps_per_decision,
        min_value=1,
        label="search-steps-per-decision",
    )

    explorers = _as_factories(definition.solvers.get("exploration"))
    exploiters = _as_factories(definition.solvers.get("exploitation"))
    if not explorers or not exploiters:
        raise SystemExit(f"Problem '{args.problem}' has no solver factories to compare")

    results: List[PairResult] = []

    # Deterministic episode seeds shared across all pairs.
    episode_seeds = [base_seed + i for i in range(episodes)]
    episode_num_cities: Optional[List[int]] = None
    if num_cities_choices is not None:
        episode_num_cities = [num_cities_choices[i % len(num_cities_choices)] for i in range(episodes)]

    total_pairs = len(explorers) * len(exploiters)
    pair_index = 0
    progress_every = _progress_every(episodes)

    for exp_factory in explorers:
        for imp_factory in exploiters:
            pair_index += 1
            print(
                f"[{pair_index}/{total_pairs}] {exp_factory.cls.__name__} x {imp_factory.cls.__name__}",
                file=sys.stderr,
                flush=True,
            )
            fitnesses: List[float] = []
            for ep_idx, ep_seed in enumerate(episode_seeds, start=1):
                if ep_idx == 1 or ep_idx == episodes or (progress_every > 0 and ep_idx % progress_every == 0):
                    print(
                        f"  episode {ep_idx}/{episodes}",
                        file=sys.stderr,
                        flush=True,
                    )
                adapter_kwargs = {"seed": int(ep_seed)}
                if episode_num_cities is not None:
                    adapter_kwargs["num_cities"] = int(episode_num_cities[ep_idx - 1])
                bundle = definition.instantiate(
                    adapter_kwargs=adapter_kwargs,
                    solver_kwargs={
                        "exploration": {"seed": int(ep_seed)},
                        "exploitation": {"seed": int(ep_seed)},
                    },
                )

                # Override the randomly chosen factories by constructing directly.
                problem_obj = bundle.problem
                exploration = exp_factory.build(problem_obj, overrides={"seed": int(ep_seed)})
                exploitation = imp_factory.build(problem_obj, overrides={"seed": int(ep_seed)})

                env = create_env(
                    problem_obj,
                    exploration,
                    exploitation,
                    max_decision_steps=max_decision_spec,
                    search_steps_per_decision=search_step_spec,
                    max_search_steps=args.max_search_steps,
                    reward_clip=1.0,
                    log_type="compare",
                    log_dir="logs",
                    session_id=None,
                    emit_init_summary=False,
                )

                obs, _ = env.reset(seed=int(ep_seed))

                done = False
                step = 0
                # Determine decision budget for policy schedule.
                max_dec = _upper(max_decision_spec)
                switch_step = max_dec // 2
                last_step = max(0, max_dec - 1)

                while not done:
                    if step == switch_step:
                        action = 1
                    elif step == last_step:
                        action = 1
                    else:
                        action = 0

                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    done = bool(terminated or truncated)
                    step += 1

                best = env.get_best_solution()
                fit = float(best.fitness) if best is not None and best.fitness is not None else float("inf")
                fitnesses.append(fit)
                env.close()

            arr = np.asarray(fitnesses, dtype=float)
            results.append(
                PairResult(
                    explorer=exp_factory.cls.__name__,
                    exploiter=imp_factory.cls.__name__,
                    episodes=int(episodes),
                    fitness_mean=float(np.mean(arr)),
                    fitness_std=float(np.std(arr)) if arr.size > 1 else 0.0,
                    fitness_min=float(np.min(arr)),
                    fitness_max=float(np.max(arr)),
                )
            )

    results.sort(key=lambda r: r.fitness_mean)

    # Print a compact ranked table.
    print(f"Problem={args.problem} episodes_per_pair={episodes} pairs={len(results)}")
    print("rank\texplorer\texploiter\tmean\tstd\tmin\tmax")
    for i, r in enumerate(results, start=1):
        print(
            f"{i}\t{r.explorer}\t{r.exploiter}\t"
            f"{r.fitness_mean:.6g}\t{r.fitness_std:.6g}\t{r.fitness_min:.6g}\t{r.fitness_max:.6g}"
        )


def _progress_every(episodes: int) -> int:
    """Choose a reasonable progress cadence for per-episode updates."""
    episodes = int(episodes)
    if episodes <= 1:
        return 0
    # Aim for ~10 updates, but avoid spamming.
    return max(1, episodes // 10)


def _upper(spec: IntRangeSpec) -> int:
    if isinstance(spec, tuple):
        return int(spec[1])
    return int(spec)


def _as_factories(spec) -> List:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return list(spec)
    return [spec]


if __name__ == "__main__":
    main()
