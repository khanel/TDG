# RL Orchestrator Framework

A clean, extensible framework for RL-controlled orchestration of exploration and exploitation solvers in optimization problems.

## Overview

This framework provides abstract interfaces for solvers and problems, allowing easy integration of external algorithms (e.g., MAP-Elites, SA, GWO) for phased search. It separates RL training (policy learning) from inference (solving), and supports any problem via adapters.

## Observation Space

The observation space is an 8-element vector in [0,1], designed to be predictive, problem-agnostic, and efficient:

1. `budget_remaining` — Normalized countdown of the remaining evaluation budget (`1 - step_ratio`).
2. `normalized_best_fitness` — Best fitness normalized to [0,1] using provided bounds (lower is better).
3. `improvement_velocity` — Smoothed (EWMA) rate of improvement in `normalized_best_fitness`.
4. `stagnation_nonparametric` — Statistical measure of progress (1 - p-value of Mann-Whitney U test on fitness history).
5. `population_concentration` — Mean distance of population from its centroid, indicating diversity loss.
6. `landscape_funnel_proxy` — Spearman correlation of neighbors' distance and fitness, probing basin smoothness.
7. `landscape_deceptiveness_proxy` — Relative fitness of a heavily mutated solution, sensing alternative basins.
8. `active_phase` — 0 for exploration, 1 for exploitation.

## Structure

- `core/`: Orchestrator wrapper, observation, reward, and utilities (problem-agnostic).
- `tsp/`: TSP adapter, solvers (`map_elites`, `pso`), and RL entry points (`rl/train.py`, `rl/evaluate.py`).
- `maxcut/`: Max-Cut adapter, solvers (`explorer`, `local_search`), and RL entry points (`rl/train.py`, `rl/evaluate.py`).
- `knapsack/`: Knapsack adapter, solvers (`explorer`, `local_search`), and RL entry points (`rl/train.py`, `rl/evaluate.py`).
- `knapsack/`: Knapsack adapter.
- `problems/registry.py`: Lightweight registry that maps problem names to adapters from the per-problem packages (kept for compatibility).
- `solvers/`: Registry for registering additional exploration/exploitation algorithms; default registrations import the implementations from the per-problem packages.
- `rl/environment.py`: Problem-agnostic Gymnasium environment used by all training scripts.

## Usage

Problem-specific training and evaluation scripts live alongside each adapter:

```bash
# Train on TSP (random instance regenerated each episode; `--tsp-num-cities` accepts ranges like 20-50)
python -m RLOrchestrator.tsp.rl.train \
  --total-timesteps 200000 \
  --tsp-num-cities 40-80 \
  --tsp-grid-size 150 \
  --progress-bar

# Evaluate a trained TSP policy and save route/timeline plots
python -m RLOrchestrator.tsp.rl.evaluate \
  --model-path ppo_tsp.zip \
  --episodes 10 \
  --output-dir outputs/tsp_eval

# Train on Max-Cut with random Erdős–Rényi graphs
python -m RLOrchestrator.maxcut.rl.train \
  --total-timesteps 150000 \
  --n-nodes 128 \
  --edge-probability 0.35 \
  --progress-bar

# Evaluate a Max-Cut policy (plots partition and fitness timeline)
python -m RLOrchestrator.maxcut.rl.evaluate \
  --model-path ppo_maxcut.zip \
  --episodes 5 \
  --output-dir outputs/maxcut_eval

# Train on Knapsack (random instance regenerated each episode unless arrays provided)
python -m RLOrchestrator.knapsack.rl.train \
  --total-timesteps 120000 \
  --n-items 60-120 \
  --value-range 1 200 \
  --weight-range 1 60 \
  --capacity-ratio 0.45 \
  --progress-bar

# Evaluate a Knapsack policy (plots selected items and fitness timeline)
python -m RLOrchestrator.knapsack.rl.evaluate \
  --model-path ppo_knapsack.zip \
  --episodes 10 \
  --output-dir outputs/knapsack_eval
```

Both the TSP and Max-Cut adapters regenerate a fresh random instance at the beginning of every episode whenever no explicit coordinates/weight matrix is supplied.

## Extending

- Implement a new adapter under `RLOrchestrator/<problem>/adapter.py` and register it in `problems/registry.py`.
- Add exploration/exploitation solvers in `RLOrchestrator/<problem>/solvers/` and register them via `solvers/registry.py`.
- Create problem-specific RL entry points under `RLOrchestrator/<problem>/rl/` that build on the shared environment.
