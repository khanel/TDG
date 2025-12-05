"""
MaxCut solver implementations for exploration and exploitation phases.

Provides 11 explorers × 13 exploiters = 143 unique pairings.

Explorer variants are configured for:
- Diversity maintenance
- Global search
- High randomness/perturbation
- Low selection pressure

Exploiter variants are configured for:
- Convergence to local optima
- Local refinement (greedy bit flips)
- Low randomness
- High selection pressure
"""

# =============================================================================
# Explorer Variants - Configured for DIVERSITY and GLOBAL SEARCH
# =============================================================================
from .explorers import (
    MaxCutMapElitesExplorer,
    MaxCutGAExplorer,
    MaxCutPSOExplorer,
    MaxCutGWOExplorer,
    MaxCutABCExplorer,
    MaxCutWOAExplorer,
    MaxCutHHOExplorer,
    MaxCutMPAExplorer,
    MaxCutSMAExplorer,
    MaxCutGSAExplorer,
    MaxCutDiversityExplorer,
)

# =============================================================================
# Exploiter Variants - Configured for CONVERGENCE and LOCAL REFINEMENT
# =============================================================================
from .exploiters import (
    MaxCutGreedyExploiter,
    MaxCutGAExploiter,
    MaxCutPSOExploiter,
    MaxCutGWOExploiter,
    MaxCutABCExploiter,
    MaxCutWOAExploiter,
    MaxCutHHOExploiter,
    MaxCutMPAExploiter,
    MaxCutSMAExploiter,
    MaxCutGSAExploiter,
    MaxCutHillClimbingExploiter,
    MaxCutMemeticExploiter,
    MaxCutLSHADEExploiter,
)

# =============================================================================
# Backward compatibility aliases
# =============================================================================
MaxCutRandomExplorer = MaxCutGAExplorer  # Similar behavior
MaxCutBitFlipExploiter = MaxCutGreedyExploiter  # Similar behavior

# Legacy hybrid imports (deprecated, kept for backward compatibility)
from .hybrid import (
    MaxCutArtificialBeeColony,
    MaxCutGravitationalSearch,
    MaxCutHarrisHawks,
    MaxCutLSHADE,
    MaxCutMarinePredators,
    MaxCutMemeticAlgorithm,
    MaxCutSlimeMould,
    MaxCutWhaleOptimization,
)

# =============================================================================
# Solver Lists for Registry
# =============================================================================
EXPLORER_CLASSES = [
    MaxCutMapElitesExplorer,
    MaxCutGAExplorer,
    MaxCutPSOExplorer,
    MaxCutGWOExplorer,
    MaxCutABCExplorer,
    MaxCutWOAExplorer,
    MaxCutHHOExplorer,
    MaxCutMPAExplorer,
    MaxCutSMAExplorer,
    MaxCutGSAExplorer,
    MaxCutDiversityExplorer,
]

EXPLOITER_CLASSES = [
    MaxCutGreedyExploiter,
    MaxCutGAExploiter,
    MaxCutPSOExploiter,
    MaxCutGWOExploiter,
    MaxCutABCExploiter,
    MaxCutWOAExploiter,
    MaxCutHHOExploiter,
    MaxCutMPAExploiter,
    MaxCutSMAExploiter,
    MaxCutGSAExploiter,
    MaxCutHillClimbingExploiter,
    MaxCutMemeticExploiter,
    MaxCutLSHADEExploiter,
]

__all__ = [
    # Explorer variants (11 total)
    "MaxCutMapElitesExplorer",
    "MaxCutGAExplorer",
    "MaxCutPSOExplorer",
    "MaxCutGWOExplorer",
    "MaxCutABCExplorer",
    "MaxCutWOAExplorer",
    "MaxCutHHOExplorer",
    "MaxCutMPAExplorer",
    "MaxCutSMAExplorer",
    "MaxCutGSAExplorer",
    "MaxCutDiversityExplorer",
    
    # Exploiter variants (13 total)
    "MaxCutGreedyExploiter",
    "MaxCutGAExploiter",
    "MaxCutPSOExploiter",
    "MaxCutGWOExploiter",
    "MaxCutABCExploiter",
    "MaxCutWOAExploiter",
    "MaxCutHHOExploiter",
    "MaxCutMPAExploiter",
    "MaxCutSMAExploiter",
    "MaxCutGSAExploiter",
    "MaxCutHillClimbingExploiter",
    "MaxCutMemeticExploiter",
    "MaxCutLSHADEExploiter",
    
    # Backward compatibility
    "MaxCutRandomExplorer",
    "MaxCutBitFlipExploiter",
    
    # Legacy hybrid classes
    "MaxCutArtificialBeeColony",
    "MaxCutGravitationalSearch",
    "MaxCutHarrisHawks",
    "MaxCutLSHADE",
    "MaxCutMarinePredators",
    "MaxCutMemeticAlgorithm",
    "MaxCutSlimeMould",
    "MaxCutWhaleOptimization",
    
    # Class lists for registry
    "EXPLORER_CLASSES",
    "EXPLOITER_CLASSES",
]
