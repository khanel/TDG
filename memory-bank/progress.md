# Progress

## What Works

-   The core architecture has been refactored into a single, integrated `OrchestratorEnv`.
-   A centralized `create_env` factory ensures consistent environment creation.
-   All training and evaluation scripts have been updated to use the new factory.
-   The project's package-level exports (`__init__.py`) are updated.

## What's Left to Build

-   Implementation of the observation and reward computers.
-   Wiring for the initial TSP use case.
-   The full training and experimentation framework.

## Current Status

-   **Phase:** Core Implementation.
-   **Next Milestone:** Define and implement the minimal observation space (Manifesto section 3).

## Known Issues

-   None.