"""
Core utilities for the RL Orchestrator Framework.
This package is now aligned with root Core/ APIs. Use Core.problem and Core.search_algorithm
as the canonical interfaces. The Orchestrator here wraps Core search algorithms.
"""

from .context import BudgetSpec, OrchestratorContext, StageBinding
from .orchestrator import OrchestratorEnv
from .observation import ObservationComputer
from .stage_controller import StageController
from .deployment import DeployableOrchestrator, ObservationConfig, PRESETS, create_deployable_agent
from .reward import RewardWrapper, EFRConfig
from .utils import *

__all__ = [
    'BudgetSpec', 'OrchestratorContext', 'StageBinding',
    'StageController', 'OrchestratorEnv',
    'ObservationComputer',
    'DeployableOrchestrator', 'ObservationConfig', 'PRESETS', 'create_deployable_agent',
    'RewardWrapper', 'EFRConfig',
]
