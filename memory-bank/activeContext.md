# Active Context

## Current Focus

We completed the architectural refactor of the RL orchestrator: `ProblemDefinition` bundles now compose adapters and solver factories, `OrchestratorContext` + `StageController` own the state machine, and every training/evaluation script builds environments via `instantiate_problem(...)`. Exploration/exploitation solvers advertise phases, and adapters now expose neighbor sampling to support observation metrics.

## Next Steps

1. **Register the full solver catalog:** Add all remaining exploration/exploitation solvers (GA, PSO variants, etc.) to the problem registry with accurate `phase` metadata so generalized training can sample the entire pool.
2. **Validate generalized training:** Exercise `rl/train_generalized.py` to confirm multi-problem policies train correctly and capture the expanded solver variety.
3. **Finish observation/reward improvements:** Introduce structured `OrchestratorState` snapshots for the observation computer, finalize diversity-collapse tracking, and retune rewards as needed.
4. **Documentation & tests:** Update README/architecture docs with the solver coverage expectations and add targeted tests (StageController, registry) to lock in the new architecture.*** End Patch
