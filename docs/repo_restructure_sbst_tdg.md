# Repo restructuring: SBST TDG integration

Goal: integrate **SBST Test Data Generation (TDG)** as a new *problem area* without relocating the whole repository.

## What changed (Dec 2025)
- Added `RLOrchestrator/sbst/` as the home for the Java SBST-TDG code.
- Moved root-level solver implementation folders into `algorithms/` (e.g., `GA/`, `GWO/`, `HHO/`, `PSO/`, `WOA/`, etc.).
- Moved root-level benchmark problem folders into `problems/` (e.g., `TSP/`, `MaxCut/`, `Knapsack/`, `NKL/`).
- Registered `sbst` in `RLOrchestrator/problems/registry.py` (scaffold adapter + placeholder solvers) so it can be instantiated like other problems.

Rationale:
- Matches the existing pattern where problems live under `RLOrchestrator/` (e.g., `tsp/`, `maxcut/`, `knapsack/`, `nkl/`).
- Avoids creating a new repo-wide root package and keeps existing solvers/framework intact.
 - Separates “algorithm implementations” (metaheuristics) from “problems” (TSP/MaxCut/Knapsack/NKL) at the repo root.

## What this does NOT include yet
- No real SBST execution pipeline yet (JUnit generation, Maven/Gradle test execution, JaCoCo XML parsing).
- No inheritance-aware coverage gating yet (parent-first, child-after-parent-complete).
- The SBST adapter currently uses a placeholder surrogate objective only to keep the solver/orchestrator integration working.

## Next restructuring steps (when ready)
- Create the SBST problem adapter that exposes a clean evaluation API for “generate tests → run → parse JaCoCo”.
- Expand the SBST registry entry from “scaffold” to “real pipeline-backed” (so the orchestrator can train/evaluate on SBST signals).
- Decide where long-running artifacts live (e.g., `results/` or a structured `runs/` folder) and how caching state is stored.
