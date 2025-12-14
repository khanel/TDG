# Repo restructuring: SBST TDG integration

Goal: integrate **SBST Test Data Generation (TDG)** as a new *problem area* without relocating the whole repository.

## What changed (Dec 2025)
- Added `RLOrchestrator/sbst/` as the home for the Java SBST-TDG code.
- Moved root-level solver implementation folders into `algorithms/` (e.g., `GA/`, `GWO/`, `HHO/`, `PSO/`, `WOA/`, etc.).
- Moved root-level benchmark problem folders into `problems/` (e.g., `TSP/`, `MaxCut/`, `Knapsack/`, `NKL/`).

Rationale:
- Matches the existing pattern where problems live under `RLOrchestrator/` (e.g., `tsp/`, `maxcut/`, `knapsack/`, `nkl/`).
- Avoids creating a new repo-wide root package and keeps existing solvers/framework intact.
 - Separates “algorithm implementations” (metaheuristics) from “problems” (TSP/MaxCut/Knapsack/NKL) at the repo root.

## What we are NOT doing yet
- No wiring into `RLOrchestrator/problems/registry.py` yet (implementation will come after the SBST adapter stabilizes).

## Next restructuring steps (when ready)
- Create the SBST problem adapter that exposes a clean evaluation API for “generate tests → run → parse JaCoCo”.
- Add an SBST entry to the problems registry (so the orchestrator can treat it like other problems).
- Decide where long-running artifacts live (e.g., `results/` or a structured `runs/` folder) and how caching state is stored.
