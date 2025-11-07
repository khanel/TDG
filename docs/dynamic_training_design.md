# Dynamic Episode Configuration Design

## 1. Overview

This document outlines the design for a dynamic training loop, as required by step 6.1 of the `meta_orchestrator_design_manifesto.md`. The goal is to train a single, general-purpose RL policy that can effectively orchestrate a variety of solver combinations across different problem types, preventing overfitting to any specific configuration.

## 2. Core Principle: Randomization

The key to learning a robust, general-purpose policy is to introduce a high degree of randomization at the start of each training episode. This forces the agent to learn the underlying *patterns* of when to switch, rather than memorizing the optimal switch time for a single, fixed setup.

At the beginning of each episode, the training script will:

1.  **Randomly select a problem type** from a predefined list (e.g., `['tsp', 'maxcut', 'knapsack']`).
2.  **For the selected problem, randomly select a valid pair of solvers:**
    *   One exploration-focused solver.
    *   One exploitation-focused solver.

## 3. Implementation Strategy

This logic will be implemented within the main training script (e.g., a new, generalized `RLOrchestrator/rl/train_generalized.py`).

### a. Problem and Solver Registries

The system will rely on the existing problem and solver registries (`problems/registry.py`, `solvers/registry.py`) to discover available components.

-   The **Problem Registry** will map problem names (e.g., `"tsp"`) to their respective adapter classes (`TSPAdapter`).
-   The **Solver Registry** will categorize solvers by their intended phase (`'exploration'` or `'exploitation'`) and the problem they are designed for.

### b. Episode Setup Workflow

The training loop will be modified as follows:

```python
# Pseudocode for the training loop

problem_types = ['tsp', 'maxcut'] # Configurable list
solver_registry = get_solver_registry()

for episode in range(num_episodes):
    # 1. Randomly select a problem
    selected_problem_name = random.choice(problem_types)
    problem_adapter = get_problem_adapter(selected_problem_name)
    problem_instance = problem_adapter.create_problem()

    # 2. Randomly select solvers for that problem
    available_explorers = solver_registry.get_solvers(problem=selected_problem_name, phase='exploration')
    available_exploiters = solver_registry.get_solvers(problem=selected_problem_name, phase='exploitation')

    ExplorationSolverClass = random.choice(available_explorers)
    ExploitationSolverClass = random.choice(available_exploiters)

    exploration_solver = ExplorationSolverClass(problem_instance)
    exploitation_solver = ExploitationSolverClass(problem_instance)

    # 3. Create the environment for this episode
    env = create_env(
        problem=problem_instance,
        exploration_solver=exploration_solver,
        exploitation_solver=exploitation_solver,
        # ... other parameters
    )

    # 4. Run the training episode on this environment
    model.learn(total_timesteps=episode_length, reset_num_timesteps=False)

```

## 4. Benefits

-   **Generalization:** The resulting policy will be robust and applicable to new, unseen combinations of problems and solvers.
-   **Extensibility:** Adding a new problem or solver automatically includes it in the training distribution, requiring no changes to the core training logic.
-   **Efficiency:** A single training run produces a policy that would otherwise require separate training runs for every possible combination.