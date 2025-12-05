"""NK-Landscape solver implementations.

This module provides properly-tuned explorer and exploiter variants of
multiple metaheuristic algorithms for binary NKL problems.

Explorer variants are configured for:
- Diversity maintenance
- Global search
- High randomness/perturbation
- Low selection pressure

Exploiter variants are configured for:
- Convergence to local optima
- Local refinement
- Low randomness
- High selection pressure
"""

# =============================================================================
# Explorer Variants - Configured for DIVERSITY and GLOBAL SEARCH
# =============================================================================
from .explorers import (
    NKLMapElitesExplorer,
    NKLGWOExplorer,
    NKLPSOExplorer,
    NKLGAExplorer,
    NKLABCExplorer,
    NKLWOAExplorer,
    NKLHHOExplorer,
    NKLMPAExplorer,
    NKLSMAExplorer,
    NKLGSAExplorer,
    NKLDiversityExplorer,
)

# =============================================================================
# Exploiter Variants - Configured for CONVERGENCE and LOCAL REFINEMENT
# =============================================================================
from .exploiters import (
    NKLBinaryPSOExploiter,
    NKLGWOExploiter,
    NKLPSOExploiter,
    NKLGAExploiter,
    NKLLSHADEExploiter,
    NKLWOAExploiter,
    NKLHHOExploiter,
    NKLMPAExploiter,
    NKLSMAExploiter,
    NKLGSAExploiter,
    NKLHillClimbingExploiter,
    NKLMemeticExploiter,
    NKLABCExploiter,
)

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    # Explorer variants (11 total)
    "NKLMapElitesExplorer",
    "NKLGWOExplorer",
    "NKLPSOExplorer",
    "NKLGAExplorer",
    "NKLABCExplorer",
    "NKLWOAExplorer",
    "NKLHHOExplorer",
    "NKLMPAExplorer",
    "NKLSMAExplorer",
    "NKLGSAExplorer",
    "NKLDiversityExplorer",
    
    # Exploiter variants (13 total)
    "NKLBinaryPSOExploiter",
    "NKLGWOExploiter",
    "NKLPSOExploiter",
    "NKLGAExploiter",
    "NKLLSHADEExploiter",
    "NKLWOAExploiter",
    "NKLHHOExploiter",
    "NKLMPAExploiter",
    "NKLSMAExploiter",
    "NKLGSAExploiter",
    "NKLHillClimbingExploiter",
    "NKLMemeticExploiter",
    "NKLABCExploiter",
    
    # Convenience lists
    "EXPLORER_CLASSES",
    "EXPLOITER_CLASSES",
]


# =============================================================================
# Convenience lists for solver factory
# =============================================================================
EXPLORER_CLASSES = [
    NKLMapElitesExplorer,
    NKLGWOExplorer,
    NKLPSOExplorer,
    NKLGAExplorer,
    NKLABCExplorer,
    NKLWOAExplorer,
    NKLHHOExplorer,
    NKLMPAExplorer,
    NKLSMAExplorer,
    NKLGSAExplorer,
    NKLDiversityExplorer,
]

EXPLOITER_CLASSES = [
    NKLBinaryPSOExploiter,
    NKLGWOExploiter,
    NKLPSOExploiter,
    NKLGAExploiter,
    NKLLSHADEExploiter,
    NKLWOAExploiter,
    NKLHHOExploiter,
    NKLMPAExploiter,
    NKLSMAExploiter,
    NKLGSAExploiter,
    NKLHillClimbingExploiter,
    NKLMemeticExploiter,
    NKLABCExploiter,
]
