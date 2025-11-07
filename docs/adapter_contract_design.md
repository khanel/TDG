# Adapter Contract Design

## 1. Overview

This document defines the contract for problem adapters, as required by step 7.1 of the `meta_orchestrator_design_manifesto.md`. Adapters are the crucial link between the problem-agnostic orchestration layer and the problem-specific details (like solver implementations and instance generation).

## 2. Core Responsibilities

Each adapter is responsible for:

1.  **Problem Instantiation:** Creating a concrete instance of the problem to be solved (e.g., a specific TSP instance with a set number of cities).
2.  **Solver Provisioning:** Providing the exploration and exploitation solvers that are compatible with that problem.
3.  **Environment Creation:** Encapsulating all of the above into a ready-to-use `OrchestratorEnv` instance.

## 3. Adapter Protocol (Pythonic Interface)

All problem adapters are expected to implement the following interface.

```python
from typing import Protocol, Type
from RLOrchestrator.core.orchestrator import OrchestratorEnv
from Core.problem import ProblemInterface
from Core.search_algorithm import SearchAlgorithm

class ProblemAdapter(Protocol):
    """
    The contract for any problem adapter compatible with the RL Orchestrator.
    """

    def create_problem(self, **kwargs) -> ProblemInterface:
        """
        Creates and returns an instance of the problem.
        The kwargs can be used to pass problem-specific parameters,
        like the number of cities for TSP.
        """
        ...

    def get_exploration_solver(self) -> Type[SearchAlgorithm]:
        """
        Returns the class of the default exploration solver for this problem.
        """
        ...

    def get_exploitation_solver(self) -> Type[SearchAlgorithm]:
        """
        Returns the class of the default exploitation solver for this problem.
        """
        ...

    def build_env(self, **kwargs) -> OrchestratorEnv:
        """
        Builds and returns a fully configured OrchestratorEnv for this problem.
        This method orchestrates the creation of the problem and solvers.
        """
        ...
```

## 4. Example Usage (in the generalized training loop)

```python
# Pseudocode from dynamic_training_design.md

# Assumes a registry that maps "tsp" to an instance of TSPAdapter
adapter = get_problem_adapter("tsp") 

# The adapter handles the details of creating the environment
env = adapter.build_env(
    tsp_num_cities=random.randint(20, 50) # Pass problem-specific params
)

model.learn(env)
```

This contract ensures that the training loop remains clean and problem-agnostic, delegating all the problem-specific setup to the adapter.