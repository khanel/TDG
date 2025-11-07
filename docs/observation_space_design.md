# Observation Space Design (Minimal)

## 1. Overview

This document outlines the design for the minimal observation space for the RL orchestrator, as required by step 3.1 of the `meta_orchestrator_design_manifesto.md`. The goal is to provide the agent with just enough information to make an informed decision about when to transition between stages, without overwhelming it with redundant or low-signal features.

## 2. Guiding Principles

-   **Minimalism:** Start with a small, essential set of features. More can be added later through experimentation.
-   **Normalization:** All features should be normalized to the range `[0, 1]` to ensure they are on a comparable scale for the RL agent.
-   **Informativeness:** Each feature should provide a clear signal about the state of the search process.

## 3. Finalized Observation Vector (6-dimensional)

The observation space is a six-element vector. Each feature is normalized to `[0, 1]` (or `[-1, 1]` where explicitly noted) to keep inputs on comparable scales for the policy.

1.  **`budget_remaining`**
    -   **Description:** Fraction of the decision budget that has not yet been consumed.
    -   **Calculation:** `1.0 - step_ratio`, where `step_ratio = decision_count / max_decision_steps`.
    -   **Signal:** Encourages time-aware policies. Low remaining budget should bias the agent toward decisive switching.

2.  **`normalized_best_fitness`**
    -   **Description:** Best-so-far fitness, normalized using problem-provided bounds.
    -   **Calculation:** `(best_fitness - lower_bound) / (upper_bound - lower_bound)` with clipping to `[0, 1]`.
    -   **Signal:** Communicates absolute solution quality irrespective of problem scale.

3.  **`improvement_velocity`**
    -   **Description:** Exponentially weighted moving average (EWMA) of the improvement rate in normalized best fitness.
    -   **Calculation:**
        ```
        delta = prev_normalized_best_fitness - normalized_best_fitness
        velocity = alpha * delta + (1 - alpha) * velocity
        ```
        with `alpha = 0.3` and clipping to `[-1, 1]`.
    -   **Signal:** Positive values indicate ongoing improvements; values near zero reveal stagnation.

4.  **`stagnation`**
    -   **Description:** Binary indicator flagging whether the best fitness has remained unchanged across the recent observation window.
    -   **Calculation:** Maintain a deque of the latest `stagnation_window = 10` fitness samples; output `1.0` if the first and last entries are identical, otherwise `0.0`.
    -   **Signal:** Alerts the agent when progress has plateaued long enough to justify switching.

5.  **`population_diversity`**
    -   **Description:** Mean distance of population members from the centroid in a unit-normalized solution space.
    -   **Calculation Steps:**
        1. Extract solution representations and stack into an array.
        2. Normalize each dimension to `[0, 1]` via min–max scaling.
        3. Compute the centroid and average Euclidean distance to it.
        4. Divide by `sqrt(dimensions) / 2` and clip to `[0, 1]`.
    -   **Signal:** High diversity signifies broad exploration; low diversity indicates convergence.

6.  **`active_phase`**
    -   **Description:** Encodes the current stage of the pipeline.
    -   **Calculation:** `0.0` for exploration, `1.0` for exploitation.
    -   **Signal:** Allows the policy to condition behavior on the active solver.

## 4. Implementation Notes

-   The `ObservationComputer` lives in `RLOrchestrator/core/observation.py` and owns the state required for velocity and stagnation tracking.
-   Problem metadata must supply meaningful fitness bounds; fallback defaults to `[0, 1]` if absent.
-   Diversity computation expects population members to expose a numeric `representation`. Algorithms that cannot supply this should implement an adapter before integration.