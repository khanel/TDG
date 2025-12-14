"""SBST explorers (placeholder set).

Mirrors the structure used by other RLOrchestrator problems:
- `explorers.py` provides explorer variants
- `exploiters.py` provides exploiter variants
- package `__init__.py` re-exports variants

For now we ship a single minimal explorer so SBST can be registered and exercised.
"""

from .explorer import SBSTRandomExplorer

__all__ = ["SBSTRandomExplorer"]
