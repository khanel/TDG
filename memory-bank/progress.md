# Progress

## What Works

-   **Core Framework:** The `OrchestratorEnv` is fully implemented, providing a stable foundation for experimentation.
-   **Component-Based Design:** The `ObservationComputer` and `RewardComputer` are implemented with a minimal, well-defined 6D feature set.
-   **Centralized Configuration:** The `create_env` factory ensures consistent and maintainable environment setup.
-   **Dynamic Discovery:** The problem and solver registries dynamically discover available components.

## What's Left to Build

-   A structured experimentation framework as defined in `docs/experiment_methodology.md`.
-   A training script for the baseline 6D policy.
-   An evaluation script to benchmark the 6D policy and collect comprehensive metrics.
-   Adapters and wiring for additional problems beyond TSP, MaxCut, and Knapsack.

## Current Status

-   **Phase:** Ready to establish the 6D baseline.
-   **Next Milestone:** Implement the necessary scripts to train and evaluate the baseline 6D policy.

## Known Issues

-   None.