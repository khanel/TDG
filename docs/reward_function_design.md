# Reward Function Design (Minimal)

## 1. Overview

This document outlines the design for the minimal reward function for the RL orchestrator, as required by step 4.1 of the `meta_orchestrator_design_manifesto.md`. The function is designed to encourage the agent to find high-quality solutions efficiently by learning the optimal time to switch from exploration to exploitation.

## 2. Guiding Principles

-   **Simplicity:** The reward signal should be easy to interpret and debug. It should directly incentivize the desired behavior.
-   **Focus on Improvement:** The primary driver for the reward should be the discovery of better solutions.
-   **Incentivize Smart Transitions:** The agent should be rewarded for switching at the right time (e.g., when exploration is no longer fruitful) and penalized for switching at the wrong time (e.g., when exploration is still making rapid progress).

## 3. Reward Components

The total reward `R` at each step is a sum of three components:

`R = R_improvement + R_action`

### a. Improvement Reward (`R_improvement`)

This component rewards the agent for making progress, defined as any improvement in the best-known fitness.

-   **Calculation:** `R_improvement = max(0, previous_best_fitness - current_best_fitness)`
-   **Normalization:** This value should be normalized by the problem's fitness range to keep it on a consistent scale.
-   **Purpose:** This is the primary, dense reward that encourages the agent to make the underlying solver work effectively.

### b. Action Reward (`R_action`)

This component provides a direct incentive or penalty based on the action taken (`STAY` or `ADVANCE`) and the context in which it was taken.

-   **`STAY` Action (0):**
    -   **Reward:** A small, constant negative reward (a "time penalty"). For example, `-0.01`.
    -   **Purpose:** This encourages the agent to be efficient. Lingering in a phase has a small but accumulating cost, pushing the agent to eventually `ADVANCE`.

-   **`ADVANCE` Action (1):**
    -   **Reward:** The reward for advancing is conditional on the state of the search.
        -   **Scenario 1: Advancing from a "stagnated" state.** A positive reward is given. For example, `+0.5`. This encourages the agent to move on when the current phase is no longer productive.
        -   **Scenario 2: Advancing from a "productive" state.** A negative reward (penalty) is given. For example, `-0.5`. This discourages the agent from prematurely abandoning a solver that is still making good progress.
    -   **"Stagnated" Definition:** A state is considered stagnated if the `stagnation` feature in the observation vector is high (e.g., > 0.8).

## 4. Implementation

This reward structure will be implemented in the `RewardComputer` class in `RLOrchestrator/core/reward.py`. The `compute` method will take the necessary inputs (action, improvement, observation) to calculate these components. The weights and thresholds (`-0.01`, `+0.5`, `-0.5`, `0.8`) will be initial values, subject to tuning during experimentation.