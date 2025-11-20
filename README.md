# TDG: RL Meta-Heuristic Orchestrator

TDG is a research platform for **problem-agnostic, solver-agnostic orchestration of population-based meta-heuristics**. The system exposes a Gym environment where an RL policy decides when to advance from an exploration solver to an exploitation solver while keeping the entire population intact. Everything revolves around three promises:

1. **Stay lean** – the agent only controls the transition timing (`STAY` vs `ADVANCE`).
2. **Stay agnostic** – problems configure adapters and solvers declaratively.
3. **Stay observable** – an ELA-driven observation space captures search dynamics without problem-specific hacks.

---

## Why RL-Guided Orchestration?

- **Meta-heuristics have phases**: explorers cover space well but converge slowly; exploiters refine basins but stall elsewhere. Hand-tuned schedules never generalize.
- **Population continuity matters**: the downstream solver should inherit every candidate generated upstream.
- **Binary decisions are learnable**: by reducing the control surface to `[STAY, ADVANCE]`, the policy can focus on detecting when the current solver is losing steam.
- **ELA features generalize**: exploratory landscape analysis (ELA) summarizes urgency, fitness altitude, dispersion, and funnel structure without referencing problem-specific encodings.

---

## System Architecture

```
ProblemDefinition --> ProblemBundle --> StageController <-> Solvers
                  \                                  /
                   \-> OrchestratorContext -> OrchestratorEnv
                                     |                 |
                                     v                 v
                            ObservationComputer   RewardComputer
```

- **Problem Definitions (`RLOrchestrator/problems/registry.py`)**: Declarative records that list the adapter, solver factories per phase, and metadata. `instantiate_problem(name, **overrides)` is the only supported entry point.
- **OrchestratorContext (`core/context.py`)**: Single source of truth for population state, budgets, RNG, and best-so-far statistics.
- **StageController (`core/stage_controller.py`)**: Enforces the uni-directional lifecycle (`Exploration → Exploitation → Termination`), handles solver initialization, and transfers populations.
- **OrchestratorEnv (`core/orchestrator.py`)**: Gym environment that wraps the controller, computes observations/rewards, and exposes `.step(action)` to RL algorithms.
- **Observation & Reward computers (`core/observation.py`, `core/reward.py`)**: Feature engineering and reward shaping live here so policies remain clean models.

---

## Solver & Problem Contracts

1. **Population-based meta-heuristics only** – every solver must operate on a whole population (e.g., MAP-Elites, PSO, GA variants) and expose representations + fitness values each step.
2. **Phase declaration** – solvers declare `phase = "exploration"` or `"exploitation"` so the registry can bind them correctly.
3. **Batch evaluation semantics** – solvers consume evaluation budget in batches, keeping the orchestrator's accounting accurate.
4. **Stateless hand-offs** – solvers accept the incoming population wholesale and mutate the shared `OrchestratorContext`. No hidden buffers or partial transfers.
5. **Problem adapters** – each optimization domain (TSP, MaxCut, Knapsack, NK-Landscape, …) contributes an adapter that wires domain data into the common orchestrator contracts.

---

## Observation & Reward Baseline

The minimal 6D observation space (described in `docs/ela_driven_observation_space.md`) strikes a balance between situational awareness and computational cost:

1. `budget_remaining` – fraction of evaluations left (`1 - current/maximum`).
2. `normalized_best_fitness` – best-so-far fitness scaled to `[0, 1]`.
3. `active_phase` – binary indicator (`0 = exploration`, `1 = exploitation`).
4. `progress_rate` – EWMA of normalized fitness improvement momentum.
5. `population_dispersion` – normalized mean distance from the population centroid.
6. `fdc_funnel_proxy` – (negative) Spearman correlation between candidate fitness and distance to the current best solution.

Rewards penalize wasted budget and pay for meaningful improvement, making premature phase switches or stagnation unprofitable.

Future experiments add one candidate feature at a time (fitness entropy, ruggedness probes, neutrality detectors, etc.) to quantify marginal gains.

---

## Experiment Methodology

1. **Baseline validation** – train PPO on the 6D observation space across multiple problems/solvers; report best fitness, switch timings, and reward distributions against heuristic schedules.
2. **Single-feature injections** – extend the observation vector by one feature, re-train with identical seeds/hyperparameters, and run statistical comparisons (e.g., Wilcoxon) against the 6D baseline.
3. **Landscape probe campaign** – enable short random-walk probes for ruggedness/neutrality/deceptiveness metrics and repeat the single-feature protocol while tracking additional budget cost.
4. **Consolidation** – promote features that deliver statistically significant, low-overhead gains into the new baseline; update docs, scripts, and registry metadata accordingly.

The full methodology (training protocol, solver pools, reporting requirements) lives in `docs/experiment_methodology.md`.

---

## Repository Layout

```
├── RLOrchestrator/           # Problem-agnostic implementation
│   ├── core/                 # Context, StageController, env factory, observation & reward logic
│   ├── problems/             # Problem registry + bundle helpers
│   ├── rl/                   # Generalized training/evaluation utilities
│   ├── tsp|maxcut|knapsack|nkl/  # Problem adapters and solver implementations
│   └── README.md             # Component-level notes
├── docs/                     # Design manifests, observation specs, experimentation plans
├── memory-bank/              # Project brief, active context, progress, research index
├── Pipfile / Pipfile.lock    # Dependency management via Pipenv
└── main.py / legacy dirs     # Historical experiments kept for reference
```

Key documents:

- `docs/architecture.md` – end-to-end architecture reference.
- `docs/ela_driven_observation_space.md` – observation philosophy, feature definitions, solver requirements.
- `docs/meta_orchestrator_design_manifesto.md` – design manifesto + roadmap checklist.
- `docs/experiment_methodology.md` – rigorous experimentation process.

---

## Getting Started

1. **Install dependencies**
   ```bash
   pipenv install --dev   # or use pip install -r requirements.txt if preferred
   ```
2. **Enter the environment**
   ```bash
   pipenv shell
   ```
3. **Run generalized training**
   ```bash
   python -m RLOrchestrator.rl.train_generalized \
     --total-timesteps 2000000 \
     --num-envs 8 \
     --model-save-path outputs/ppo_generalized.zip
   ```
   The script samples problem definitions at random, instantiates their stage bindings, and trains a single PPO policy across them.
4. **Evaluate / debug**
   - Use `python -m RLOrchestrator.rl.evaluation.<script>` variants for targeted rollouts.
   - Inspect logs under `logs/` to verify reward curves and phase timings.

When writing custom scripts, always request solvers via:

```python
from RLOrchestrator.problems.registry import instantiate_problem
bundle = instantiate_problem("tsp")
env = create_env(
    problem=bundle.problem,
    exploration_solver=bundle.stages[0].solver,
    exploitation_solver=bundle.stages[1].solver,
)
```

This keeps wiring consistent with the registry contracts.

---

## Current Focus & Next Steps

1. **Register the complete solver catalog** for every problem so generalized training can sample GA, PSO, GWO, MA, etc., with accurate `phase` metadata.
2. **Validate generalized training** by running long PPO sessions over the expanded solver pool and comparing policies to heuristic schedules.
3. **Finish observation & reward upgrades** – structured `OrchestratorState` snapshots, diversity-collapse tracking, and reward retuning.
4. **Documentation & testing** – keep README + docs aligned, and add unit tests for the StageController and registries to lock in the architecture.

Contributions that move any of the above forward are welcome—just ensure new solvers honor the population-based contract and document experimental changes thoroughly.
