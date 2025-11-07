# Tech Context

## Core Technologies

-   **Python:** The primary programming language.
-   **OpenAI Gym (`gym`):** The framework for building the RL environment.
-   **Reinforcement Learning Library (e.g., Stable Baselines3):** A library will be needed for training the RL agent. The specific choice is yet to be finalized but will be compatible with `gym`.

## Project Structure

The project follows a modular structure, with a clear separation of concerns.

-   `RLOrchestrator/core/`: Contains the central, problem-agnostic components of the system.
    -   `orchestrator.py`: Home of the `OrchestratorEnv`.
    -   `observation.py`: Home of the `ObservationComputer`.
    -   `reward.py`: Home of the `RewardComputer`.
-   `RLOrchestrator/<problem>/`: Each supported optimization problem (e.g., `tsp`, `maxcut`) has its own module.
    -   `adapter.py`: Contains the "wiring" to connect the problem-specific solvers and configuration to the generic `OrchestratorEnv`.
    -   `solvers/`: Contains the implementations of the population-based solvers for that problem.

## Key Dependencies

-   The system relies on existing, population-based meta-heuristic solvers. The initial implementation for TSP will use:
    -   **Exploration:** `TSPMapElites`
    -   **Exploitation:** `TSPParticleSwarm`