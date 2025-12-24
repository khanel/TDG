"""Run a single SBST orchestrator episode (no RL policy required).

This is a lightweight Stage-8 integration entry point that exercises:
- registry wiring (instantiate_problem)
- OrchestratorEnv loop
- SBST evaluation pipeline via solvers

It is expected to be slow on real SUTs because each evaluation may invoke Maven/Gradle.
"""

from __future__ import annotations

import argparse
import time

from ...core.env_factory import create_env
from ...core.utils import parse_int_range, setup_logging
from ...problems.registry import instantiate_problem


def _stage_map(stages):
    mapping = {binding.name: binding.solver for binding in stages}
    missing = {"exploration", "exploitation"} - mapping.keys()
    if missing:
        raise ValueError(f"Problem bundle missing stages: {sorted(missing)}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=False, default=None)
    parser.add_argument("--build-tool", choices=["auto", "maven", "gradle"], default="auto")
    parser.add_argument("--targets", type=str, nargs="*", default=[])
    parser.add_argument("--work-dir", type=str, default="runs/sbst")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=300)

    parser.add_argument("--exploration-population", type=int, default=12)
    parser.add_argument("--exploitation-population", type=int, default=8)
    parser.add_argument("--max-decisions", type=str, default="20")
    parser.add_argument("--search-steps-per-decision", type=str, default="1")
    parser.add_argument("--max-search-steps", type=int, default=None)

    parser.add_argument("--policy", choices=["stay", "switch_half"], default="switch_half")
    args = parser.parse_args()

    session_id = int(time.time())
    logger = setup_logging("episode", "sbst", session_id=session_id)
    logger.info(f"Starting SBST run_episode with args: {args}")

    adapter_kwargs = {
        "seed": args.seed,
        "project_root": args.project_root,
        "build_tool": args.build_tool,
        "targets": list(args.targets),
        "work_dir": args.work_dir,
        "timeout_seconds": int(args.timeout_seconds),
    }
    solver_overrides = {
        "exploration": {"population_size": max(1, int(args.exploration_population)), "seed": args.seed},
        "exploitation": {"population_size": max(1, int(args.exploitation_population)), "seed": args.seed},
    }

    bundle = instantiate_problem("sbst", adapter_kwargs=adapter_kwargs, solver_kwargs=solver_overrides)
    stage_map = _stage_map(bundle.stages)
    for solver in stage_map.values():
        if hasattr(solver, "initialize"):
            solver.initialize()

    max_decision_spec = parse_int_range(args.max_decisions, min_value=1, label="max-decisions")
    search_step_spec = parse_int_range(args.search_steps_per_decision, min_value=1, label="search-steps-per-decision")
    env = create_env(
        bundle.problem,
        stage_map["exploration"],
        stage_map["exploitation"],
        max_decision_steps=max_decision_spec,
        search_steps_per_decision=search_step_spec,
        max_search_steps=args.max_search_steps,
        log_type="episode",
        log_dir="logs",
        session_id=session_id,
        emit_init_summary=True,
    )

    obs, _ = env.reset(seed=args.seed)
    done = False
    total_reward = 0.0
    steps = 0

    # Simple, deterministic baseline policy:
    # - stay for the first half of decision budget, then advance once.
    max_decisions = max(1, int(max_decision_spec[1] if isinstance(max_decision_spec, tuple) else max_decision_spec))
    switch_step = max_decisions // 2

    while not done:
        if args.policy == "stay":
            action = 0
        else:
            action = 1 if steps == switch_step else 0

        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps += 1

        best = env.get_best_solution()
        best_fit = float(best.fitness) if best is not None and best.fitness is not None else None
        if steps % 5 == 0 or done:
            logger.info(f"Step={steps} action={action} reward={float(reward):.4f} best_fitness={best_fit}")

    logger.info(f"Episode finished: steps={steps} total_reward={total_reward:.4f}")
    env.close()


if __name__ == "__main__":
    main()
