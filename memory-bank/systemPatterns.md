# System Patterns

## Core Architecture

The system is designed as a uni-directional pipeline managed by a Reinforcement Learning (RL) agent.

1.  **State Machine:** The core of the system is a simple state machine that progresses through a fixed sequence of stages: `Exploration` → `Exploitation` → `Termination`. The agent cannot revisit a previous stage.

2.  **Population-Based Solvers:** Each stage is associated with a population-based meta-heuristic solver. This is a critical constraint, as the entire population of solutions must be transferred from the solver of one stage to the next.

3.  **RL-driven Transitions:** An RL agent decides *when* to transition from one stage to the next. The action space is binary: `[STAY, ADVANCE]`.

## Component Design

1.  **Centralized State:** A single, core object serves as the source of truth, holding the problem definition, solver instances, and current population. This avoids state drift and ensures all components operate on consistent data.

2.  **Protocol-Based Communication:** Interfaces between components (e.g., observation computer, reward computer, environment) are defined by strict data contracts (protocols). This enforces cohesion and prevents the use of ad-hoc data structures.

3.  **Gym Environment:** The orchestration logic is encapsulated within an `gym.Env` (OpenAI Gym environment). This provides a standard interface for training RL agents and separates the orchestration logic from the underlying solvers.