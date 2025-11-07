# Observation Space Design (Minimal)

## 1. Overview

This document outlines the design for the minimal observation space for the RL orchestrator, as required by step 3.1 of the `meta_orchestrator_design_manifesto.md`. The goal is to provide the agent with just enough information to make an informed decision about when to transition between stages, without overwhelming it with redundant or low-signal features.

## 2. Guiding Principles

-   **Minimalism:** Start with a small, essential set of features. More can be added later through experimentation.
-   **Normalization:** All features should be normalized to the range `[0, 1]` to ensure they are on a comparable scale for the RL agent.
-   **Informativeness:** Each feature should provide a clear signal about the state of the search process.

## 3. Proposed Observation Vector (6-dimensional)

The initial observation space will be a 6-dimensional vector. The existing `ObservationComputer` already defines an 8-dimensional space; this represents a refinement and simplification of that initial idea.

1.  **`budget_remaining` (Normalized):**
    -   **Description:** The fraction of the total decision-making budget that is left.
    -   **Calculation:** `1.0 - (current_decision_step / max_decision_steps)`
    -   **Signal:** Tells the agent how much time it has left. An agent might learn to be more aggressive with switching as the budget depletes.

2.  **`normalized_best_fitness`:**
    -   **Description:** The fitness of the best solution found so far, scaled to `[0, 1]`.
    -   **Calculation:** Requires problem-specific bounds (e.g., min/max possible fitness). `(current_fitness - min_fitness) / (max_fitness - min_fitness)`.
    -   **Signal:** Indicates the absolute quality of the current best solution.

3.  **`improvement_velocity`:**
    -   **Description:** The rate of improvement in fitness over a recent window of steps.
    -   **Calculation:** A measure like `(fitness_at_t - fitness_at_t-N) / N`, normalized.
    -   **Signal:** A high velocity suggests the current solver is still effective. A low or zero velocity indicates stagnation.

4.  **`stagnation`:**
    -   **Description:** The number of decision steps since the last improvement in the best-known fitness, normalized by the total budget.
    -   **Calculation:** `(steps_since_last_improvement / max_decision_steps)`
    -   **Signal:** A direct measure of how long the search has been "stuck."

5.  **`population_diversity`:**
    -   **Description:** A measure of how diverse the solutions in the current population are.
    -   **Calculation:** This could be the average pairwise distance between solutions in the population, normalized.
    -   **Signal:** In exploration, high diversity is good. In exploitation, diversity is expected to decrease as the population converges on a solution. A sudden drop in diversity during exploration might signal that it's time to switch.

6.  **`active_phase`:**
    -   **Description:** A binary indicator of the current phase.
    -   **Calculation:** `0.0` for exploration, `1.0` for exploitation.
    -   **Signal:** Ensures the agent's policy can be conditional on the current stage of the search.

## 4. Implementation

This observation space will be implemented within the `ObservationComputer` class in `RLOrchestrator/core/observation.py`. The existing 8-dimensional space will be replaced with this more focused 6-dimensional version.