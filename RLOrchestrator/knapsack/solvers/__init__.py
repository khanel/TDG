"""
Knapsack solver implementations for exploration and exploitation phases.

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
    KnapsackMapElitesExplorer,
    KnapsackGAExplorer,
    KnapsackPSOExplorer,
    KnapsackGWOExplorer,
    KnapsackABCExplorer,
    KnapsackWOAExplorer,
    KnapsackHHOExplorer,
    KnapsackMPAExplorer,
    KnapsackSMAExplorer,
    KnapsackGSAExplorer,
    KnapsackDiversityExplorer,
)

# =============================================================================
# Exploiter Variants - Configured for CONVERGENCE and LOCAL REFINEMENT
# =============================================================================
from .exploiters import (
    KnapsackGreedyExploiter,
    KnapsackGAExploiter,
    KnapsackPSOExploiter,
    KnapsackGWOExploiter,
    KnapsackABCExploiter,
    KnapsackWOAExploiter,
    KnapsackHHOExploiter,
    KnapsackMPAExploiter,
    KnapsackSMAExploiter,
    KnapsackGSAExploiter,
    KnapsackHillClimbingExploiter,
    KnapsackMemeticExploiter,
    KnapsackLSHADEExploiter,
)

# =============================================================================
# Backward compatibility aliases
# =============================================================================
KnapsackRandomExplorer = KnapsackGAExplorer  # Similar behavior
KnapsackBitFlipExploiter = KnapsackGreedyExploiter  # Similar behavior

# =============================================================================
# Solver Lists for Registry
# =============================================================================
EXPLORER_CLASSES = [
    KnapsackMapElitesExplorer,
    KnapsackGAExplorer,
    KnapsackPSOExplorer,
    KnapsackGWOExplorer,
    KnapsackABCExplorer,
    KnapsackWOAExplorer,
    KnapsackHHOExplorer,
    KnapsackMPAExplorer,
    KnapsackSMAExplorer,
    KnapsackGSAExplorer,
    KnapsackDiversityExplorer,
]

EXPLOITER_CLASSES = [
    KnapsackGreedyExploiter,
    KnapsackGAExploiter,
    KnapsackPSOExploiter,
    KnapsackGWOExploiter,
    KnapsackABCExploiter,
    KnapsackWOAExploiter,
    KnapsackHHOExploiter,
    KnapsackMPAExploiter,
    KnapsackSMAExploiter,
    KnapsackGSAExploiter,
    KnapsackHillClimbingExploiter,
    KnapsackMemeticExploiter,
    KnapsackLSHADEExploiter,
]

__all__ = [
    # Explorer variants (11 total)
    "KnapsackMapElitesExplorer",
    "KnapsackGAExplorer",
    "KnapsackPSOExplorer",
    "KnapsackGWOExplorer",
    "KnapsackABCExplorer",
    "KnapsackWOAExplorer",
    "KnapsackHHOExplorer",
    "KnapsackMPAExplorer",
    "KnapsackSMAExplorer",
    "KnapsackGSAExplorer",
    "KnapsackDiversityExplorer",
    
    # Exploiter variants (13 total)
    "KnapsackGreedyExploiter",
    "KnapsackGAExploiter",
    "KnapsackPSOExploiter",
    "KnapsackGWOExploiter",
    "KnapsackABCExploiter",
    "KnapsackWOAExploiter",
    "KnapsackHHOExploiter",
    "KnapsackMPAExploiter",
    "KnapsackSMAExploiter",
    "KnapsackGSAExploiter",
    "KnapsackHillClimbingExploiter",
    "KnapsackMemeticExploiter",
    "KnapsackLSHADEExploiter",
    
    # Backward compatibility
    "KnapsackRandomExplorer",
    "KnapsackBitFlipExploiter",
    
    # Class lists for registry
    "EXPLORER_CLASSES",
    "EXPLOITER_CLASSES",
]
