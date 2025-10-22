"""
Deprecated evaluation entry point.
Use problem-specific modules (e.g. `RLOrchestrator.tsp.rl.evaluate`).
"""

raise RuntimeError(
    "`RLOrchestrator.rl.evaluation.run` has been replaced by problem-specific "
    "scripts such as `python -m RLOrchestrator.tsp.rl.evaluate` or "
    "`python -m RLOrchestrator.maxcut.rl.evaluate`."
)
