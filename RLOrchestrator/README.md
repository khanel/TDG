# RL Orchestrator Framework

A clean, extensible framework for RL-controlled orchestration of exploration and exploitation solvers in optimization problems.

## Overview

This framework provides abstract interfaces for solvers and problems, allowing easy integration of external algorithms (e.g., MAP-Elites, SA, GWO) for phased search. It separates RL training (policy learning) from inference (solving), and supports any problem via adapters.

## Observation Space

The observation space is a 7-element vector in [0,1], designed to be problem-agnostic and efficient:

1. `phase_is_exploitation` — 0 for exploration, 1 for exploitation.
2. `normalized_best_fitness` — best fitness normalized to [0,1] with provided bounds (lower is better).
3. `frontier_improvement_flag` — 1 if the elite frontier (top‑10% cutoff) significantly improved this step (EWMA‑gated).
4. `frontier_success_rate` — fraction of recent steps (window W) with frontier_improvement_flag = 1.
5. `elite_turnover_entropy` — normalized Shannon entropy of elite IDs over recent snapshots.
6. `frontier_stagnation_ratio` — steps since last frontier improvement divided by W (capped at 1).
7. `budget_used_ratio` — fraction of episode steps consumed (step_count / max_steps).

Performance notes:
- Top‑10% elites are refreshed every T steps (no full sort; heap selection), with O(log K) incremental updates between refreshes using only the current best. Entropy updates on refresh and is cached.

## Structure

- `core/`: Orchestrator wrapper, observation, reward, and utilities (uses root Core APIs).
- `solvers/`: Registry for external solver classes (no implementations here).
- `problems/`: Adapters for specific problems (TSP, Knapsack).
- `rl/`: RL environment and training/inference scripts.

## Usage

1. Register external solvers (Core-compatible `SearchAlgorithm` classes) in `solvers/registry.py`.
2. Use problem adapters (implementing `Core.problem.ProblemInterface`) from `problems/registry.py`.
3. Train RL policy with `rl/training/train.py`.
4. Solve problems with `rl/inference/solver.py`.

## Extending

- Add new solvers by implementing `Core.search_algorithm.SearchAlgorithm` and registering.
- Add new problems by creating adapters implementing `Core.problem.ProblemInterface`.
