"""Java project discovery + inheritance hierarchy discovery."""

from .discover import DiscoveryError, discover_project, resolve_project_root

__all__ = ["discover_project", "DiscoveryError", "resolve_project_root"]
