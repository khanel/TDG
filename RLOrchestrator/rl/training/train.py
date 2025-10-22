"""
Deprecated entry point kept for backward compatibility.
Redirect users to problem-specific scripts (e.g. `RLOrchestrator.tsp.rl.train`).
"""

raise RuntimeError(
    "`RLOrchestrator.rl.training.train` is deprecated. "
    "Use problem-specific modules such as `python -m RLOrchestrator.tsp.rl.train` "
    "or `python -m RLOrchestrator.maxcut.rl.train`."
)
