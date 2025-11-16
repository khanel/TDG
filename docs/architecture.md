"""
High-level architecture reference for the RL Orchestrator framework.
"""

# RL Orchestrator Architecture

## Overview

The system exposes a problem-agnostic RL environment that learns when to transition
between exploration and exploitation solvers. Architecture goals:

- **Plug-in problems**: Each optimization problem provides its adapter and solver
  factories via `ProblemDefinition`.
- **Consistent orchestration**: `OrchestratorContext` + `StageController`
  centralize the state machine logic.
- **Lean RL env**: `OrchestratorEnv` delegates to the stage controller and focuses on the Gym API.
- **Registry-driven wiring**: Training/evaluation scripts instantiate
  everything through `instantiate_problem(...)` rather than importing solver modules
  directly.

## Component Diagram

```
ProblemDefinition -> ProblemBundle (problem instance + StageBinding[])
               |                                 |
               v                                 v
     OrchestratorContext <-- StageController --> Solvers
               |
               v
          OrchestratorEnv
               |
               v
        Observation/Reward computers
```

## Core Components

### Problem Definitions & Bundles

- Defined in `RLOrchestrator/problems/registry.py`.
- A `ProblemDefinition` declares:
  - Adapter class + default kwargs.
  - Stage-to-solver factories (`SolverFactory`) for exploration/exploitation.
  - Optional metadata for downstream tooling.
- `instantiate_problem(name, adapter_kwargs, solver_kwargs)` returns a `ProblemBundle`
  that includes the configured adapter and a list of `StageBinding`s.
- Scripts should *always* request bundles via this API to ensure wiring stays consistent.

### Context & Stage Controller

- `OrchestratorContext` (`RLOrchestrator/core/context.py`) holds mutable episode state:
  problem instance, solver bindings, RNG, budgets, counters, and best solution.
- `StageController` (`RLOrchestrator/core/stage_controller.py`) enforces the uni-directional
  state machine, handles population transfers, and samples budgets. It exposes a
  simple `step(action)` API used by the RL environment as well as scripted evaluations.

### Environment

- `OrchestratorEnv` (`RLOrchestrator/core/orchestrator.py`) wraps the controller in a Gym API.
- Responsibilities:
  - Manage observation/reward computers (`ObservationComputer`, `RewardComputer`).
  - Forward actions to the stage controller, compute rewards, and expose the best solution
    for logging/evaluation.
  - Provide convenience getters (`get_phase`, `get_best_solution`).
- The environment deliberately avoids direct references to solver-specific details—the
  controller handles progress, and observation/reward components consume solver snapshots
  via helper methods.

### Observation & Reward

- `ObservationComputer` aggregates features from the active solver (e.g., diversity,
  best fitness, budget usage). Future work will replace direct solver references with
  structured state snapshots.
- `RewardComputer` generates rewards using improvement measurements, penalizing inefficient
  transitions or early termination.

## Training/Evaluation Flow

1. CLI parses problem-specific overrides (e.g., number of cities, graph size).
2. Call `instantiate_problem` with adapter/solver overrides to obtain a `ProblemBundle`.
3. Initialize solvers (call `initialize` when available).
4. Create an `OrchestratorEnv` via `create_env(...)` using the bundle’s problem + stage solvers.
5. Train/evaluate a policy using Stable Baselines3 (or any Gym-compatible agent).

The generalized training script (`RLOrchestrator/rl/train_generalized.py`) follows the same
pattern but samples problems and solver bundles randomly per environment to encourage
policy generalization.

## Future Enhancements

- Replace ad-hoc solver snapshots in `ObservationComputer` with a typed `OrchestratorState`.
- Extend `ProblemDefinition` metadata (e.g., observation feature compatibility, reward
  scaling hints) so scripts can adapt automatically to new problem types.
- Add unit tests for `StageController`, `ProblemDefinition`, and registry helpers to
  prevent regressions as the framework grows.
