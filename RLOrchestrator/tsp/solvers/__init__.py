"""
TSP solver implementations for exploration and exploitation phases.

Provides 11 explorers × 13 exploiters = 143 unique pairings.

Explorer variants are configured for:
- Diversity maintenance
- Global search
- High randomness/perturbation
- Low selection pressure

Exploiter variants are configured for:
- Convergence to local optima
- Local refinement (2-opt, 3-opt)
- Low randomness
- High selection pressure
"""

# =============================================================================
# Explorer Variants - Configured for DIVERSITY and GLOBAL SEARCH
# =============================================================================
from .explorers import (
    TSPMapElitesExplorer,
    TSPGAExplorer,
    TSPPSOExplorer,
    TSPGWOExplorer,
    TSPABCExplorer,
    TSPWOAExplorer,
    TSPHHOExplorer,
    TSPMPAExplorer,
    TSPSMAExplorer,
    TSPGSAExplorer,
    TSPDiversityExplorer,
)

# =============================================================================
# Exploiter Variants - Configured for CONVERGENCE and LOCAL REFINEMENT
# =============================================================================
from .exploiters import (
    TSP2OptExploiter,
    TSPGAExploiter,
    TSPPSOExploiter,
    TSPGWOExploiter,
    TSPABCExploiter,
    TSPWOAExploiter,
    TSPHHOExploiter,
    TSPMPAExploiter,
    TSPSMAExploiter,
    TSPGSAExploiter,
    TSPHillClimbingExploiter,
    TSPMemeticExploiter,
    TSPLSHADEExploiter,
)

# =============================================================================
# Solver Lists for Registry
# =============================================================================
EXPLORER_CLASSES = [
    TSPMapElitesExplorer,
    TSPGAExplorer,
    TSPPSOExplorer,
    TSPGWOExplorer,
    TSPABCExplorer,
    TSPWOAExplorer,
    TSPHHOExplorer,
    TSPMPAExplorer,
    TSPSMAExplorer,
    TSPGSAExplorer,
    TSPDiversityExplorer,
]

EXPLOITER_CLASSES = [
    TSP2OptExploiter,
    TSPGAExploiter,
    TSPPSOExploiter,
    TSPGWOExploiter,
    TSPABCExploiter,
    TSPWOAExploiter,
    TSPHHOExploiter,
    TSPMPAExploiter,
    TSPSMAExploiter,
    TSPGSAExploiter,
    TSPHillClimbingExploiter,
    TSPMemeticExploiter,
    TSPLSHADEExploiter,
]

__all__ = [
    # Explorer variants (11 total)
    "TSPMapElitesExplorer",
    "TSPGAExplorer",
    "TSPPSOExplorer",
    "TSPGWOExplorer",
    "TSPABCExplorer",
    "TSPWOAExplorer",
    "TSPHHOExplorer",
    "TSPMPAExplorer",
    "TSPSMAExplorer",
    "TSPGSAExplorer",
    "TSPDiversityExplorer",
    
    # Exploiter variants (13 total)
    "TSP2OptExploiter",
    "TSPGAExploiter",
    "TSPPSOExploiter",
    "TSPGWOExploiter",
    "TSPABCExploiter",
    "TSPWOAExploiter",
    "TSPHHOExploiter",
    "TSPMPAExploiter",
    "TSPSMAExploiter",
    "TSPGSAExploiter",
    "TSPHillClimbingExploiter",
    "TSPMemeticExploiter",
    "TSPLSHADEExploiter",
    
    # Class lists for registry
    "EXPLORER_CLASSES",
    "EXPLOITER_CLASSES",
]
