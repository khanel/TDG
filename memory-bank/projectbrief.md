# Project Brief

## Goal

Design a problem-agnostic RL orchestrator that learns to manage a sequence of solvers. The system will:

- Utilize specialized, population-based meta-heuristic solvers for distinct stages of the search process (e.g., "exploration" and "exploitation").
- Progress uni-directionally: `Stage 1 (Exploration)` → `Stage 2 (Exploitation)` → `Termination`.
- Learn a policy over a binary action space: `[STAY, ADVANCE]`. The only decision is when to transition to the next stage.
- Remain lean and focused on learning the transition timing, not the search process itself.