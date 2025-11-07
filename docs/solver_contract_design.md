# Solver Contract Design

## 1. Overview

This document defines the contract for population-based solvers used within the meta-orchestrator, as required by step 1.2 of the `meta_orchestrator_design_manifesto.md`. A strict contract is essential for the orchestrator to manage different solvers in a generic, problem-agnostic way and to facilitate the transfer of populations between stages.

## 2. Core Principles

-   **Stateless Interaction:** From the orchestrator's perspective, the solver should be treated as a stateful object that can be stepped forward, but the interaction itself is simple. The orchestrator calls `run()`, and then inspects the results.
-   **Population Transfer:** The contract must explicitly support initializing a solver with an existing population. This is the core mechanism for the `EXPLORATION` -> `EXPLOITATION` transition.
-   **Problem Agnostic:** The contract should not be tied to any specific problem like TSP or MaxCut. It should rely on a generic `Problem` definition.

## 3. Solver Protocol (Pythonic Interface)

All solvers are expected to implement the following interface. While Python doesn't have formal interfaces, this will be enforced through convention and abstract base classes where appropriate.

```python
from typing import Protocol, List, Any

class Problem(Protocol):
    """A protocol defining the problem to be solved."""
    # ... problem-specific methods and properties

class Solver(Protocol):
    """
    The contract for any population-based solver compatible with the RL Orchestrator.
    """

    def __init__(self, problem: Problem, initial_population: List[Any] = None):
        """
        Initializes the solver.

        Args:
            problem: An instance of a class that adheres to the Problem protocol.
            initial_population: An optional list of solutions to start the search with.
                                This is critical for the population transfer between stages.
        """
        ...

    def run(self, iterations: int):
        """
        Runs the solver for a specified number of iterations or steps.
        This method contains the main optimization loop of the algorithm.
        """
        ...

    @property
    def population(self) -> List[Any]:
        """
        Returns the entire current population of solutions.
        This is used by the orchestrator to transfer the state to the next solver.
        """
        ...

    @property
    def best_solution(self) -> Any:
        """
        Returns the best solution found so far by the solver.
        """
        ...

    @property
    def best_fitness(self) -> float:
        """
        Returns the fitness of the best solution found so far.
        """
        ...
```

## 4. Usage in the Orchestrator

The `OrchestratorEnv` will interact with solvers as follows:

1.  **Instantiation (EXPLORATION):**
    ```python
    # At the start of an episode
    explorer = ExplorationSolver(problem)
    ```

2.  **Running the Solver:**
    ```python
    # In the STAY action
    current_solver.run(iterations=N_STEPS_PER_CALL)
    ```

3.  **Transition (ADVANCE):**
    ```python
    # Agent chose ADVANCE from EXPLORATION state
    current_population = explorer.population
    exploiter = ExploitationSolver(problem, initial_population=current_population)
    # The 'exploiter' is now the current_solver
    ```

This contract ensures a clean separation of concerns and enables the "plug-and-play" architecture envisioned for the orchestrator.