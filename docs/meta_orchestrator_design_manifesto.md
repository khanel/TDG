# Meta-Orchestrator Design Manifesto (v2 - Integrated)

## Goal
Design a problem-agnostic RL orchestrator that learns to manage a sequence of solvers. The system will:
- Utilize specialized, population-based meta-heuristic solvers for distinct stages of the search process (e.g., "exploration" and "exploitation").
- Progress uni-directionally: `Stage 1 (Exploration)` → `Stage 2 (Exploitation)` → `Termination`.
- Learn a policy over a binary action space: `[STAY, ADVANCE]`. The only decision is when to transition to the next stage.
- Remain lean and focused on learning the transition timing, not the search process itself.

## Legend
- [ ] Not Designed / Not Implemented

This document outlines the proposed design and serves as a checklist for development.

---

## 1. Core Minimal Design

[ ] **1.1 Uni-directional Pipeline:** Formalize the state machine for the `Exploration` -> `Exploitation` -> `Termination` lifecycle.
[ ] **1.2 Population-Based Solvers:** Define a clear contract for how solvers are used in each stage. Solvers must be population-based meta-heuristics to support population transfer.
[ ] **1.3 Integrated Design:** Refactor the existing `RLOrchestrator` core to support this model, ensuring seamless integration with the current codebase.

---

## 2. Interfaces and Cohesion

[ ] **2.1 Centralized State Management:** Use a core object as the single source of truth for the problem definition and solver instances.
[ ] **2.2 Protocol Contracts:** Define the data structures for communication (e.g., observations, actions, rewards) in a centralized, shared location.
[ ] **2.3 Strict Contract Adherence:** Ensure the environment and all related components use the defined contracts without ad-hoc data types.

---

## 3. Observation Space (Minimal)

[ ] **3.1 Minimal Observation Definition:** Define the input vector for the policy.
    - *Initial proposal: A 6-dimensional vector including metrics like (time_since_last_improvement, fraction_of_budget_used, etc.).*
[ ] **3.2 Observation Computer:** Design a component to compute the observation vector from the core system state.
    - *Proposed name: `ObservationComputer` to be integrated into `RLOrchestrator/core/observation.py`.*

---

## 4. Reward Function (Minimal)

[ ] **4.1 Minimal Reward Definition:** Design a reward signal to encourage efficient stage transitions.
[ ] **4.2 Reward Computer:** Implement the logic for calculating the reward.
    - *Proposed name: `RewardComputer` to be integrated into `RLOrchestrator/core/reward.py`.*
[ ] **4.3 Weight Tuning:** Plan for empirical tuning of reward components after the initial implementation.

---

## 5. Environment & Stage Transition

[ ] **5.1 Transition Semantics:** Formalize the `STAY` (0) and `ADVANCE` (1) actions. The `ADVANCE` action must trigger the transfer of the entire population from the current solver to the next.
[ ] **5.2 Orchestrator Environment:** Implement the `gym.Env` that encapsulates the orchestration logic.
    - *Proposed name: `OrchestratorEnv` to be integrated into `RLOrchestrator/core/orchestrator.py`.*

---

## 6. Generalized Training

[ ] **6.1 Dynamic Episode Configuration:** Design the training loop to be highly variable.
    - At the start of each episode, randomly select a problem type (e.g., TSP, MaxCut).
    - For the selected problem, randomly select a valid pair of configured exploration and exploitation solvers.
[ ] **6.2 Unbiased Policy Learning:** This randomization is critical to ensure the agent learns a general-purpose policy that is not overfitted to a specific problem or solver combination.

---

## 7. Problem Adapters & Wiring

[ ] **7.1 Adapter Contract:** Define how a specific problem (like TSP) provides solvers and configuration for the environment.
[ ] **7.2 Initial TSP Wiring:** Create an adapter for the Traveling Salesperson Problem as the first use case.
    - *Proposed location for new wiring logic: `RLOrchestrator/tsp/env.py`*
    - *Proposed `build_tsp_env` function:*
        - Exploration Solver: `TSPMapElites` (Population-Based)
        - Exploitation Solver: `TSPParticleSwarm` (Population-Based)
        - Wrapper: `OrchestratorEnv`
[ ] **7.3 Future Problem Wiring:** Plan for extending the framework to other problems (e.g., MaxCut, Knapsack, NKL).

---

## 8. Experimentation Framework

[ ] **8.1 Baseline Implementation:** Implement the core stack with TSP wiring to establish a performance baseline.
[ ] **8.2 Incremental Experiments:** Design a framework to toggle additional observation features or reward variants to measure their impact.

---

## 9. Naming and Structure (Integrated Approach)

[ ] **9.1 Refactoring Plan:**
    - `RLOrchestrator/core/orchestrator.py`: Will be extended to include the `OrchestratorEnv` logic.
    - `RLOrchestrator/core/observation.py`: Will be extended to include the `ObservationComputer`.
    - `RLOrchestrator/core/reward.py`: Will be extended to include the `RewardComputer`.
    - `RLOrchestrator/tsp/env.py`: New file containing the TSP-specific wiring for the environment.
[ ] **9.2 Future Refactoring:** Plan to revisit naming and structure once the design stabilizes.

---

## 10. Validation

[ ] **10.1 Validation Strategy:** Define experiments to compare the learned policy against fixed baselines (e.g., always-explore, 50/50 split).
[ ] **10.2 Documentation:** Document the final design, its scope, and its known limitations upon completion.