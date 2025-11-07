"""
Factory function for creating the OrchestratorEnv.
This centralizes environment creation for consistency in training and evaluation.
"""

from typing import Optional
from .orchestrator import OrchestratorEnv
from Core.problem import ProblemInterface
from Core.search_algorithm import SearchAlgorithm
from .utils import IntRangeSpec


def create_env(
    problem: ProblemInterface,
    exploration_solver: SearchAlgorithm,
    exploitation_solver: SearchAlgorithm,
    max_decision_steps: IntRangeSpec = 100,
    *,
    search_steps_per_decision: IntRangeSpec = 1,
    max_search_steps: Optional[int] = None,
    reward_clip: float = 1.0,
    log_type: str = 'train',
    log_dir: str = 'logs',
    session_id: Optional[int] = None,
    emit_init_summary: bool = True,
) -> OrchestratorEnv:
    """
    Factory function to create and configure an OrchestratorEnv instance.
    """
    return OrchestratorEnv(
        problem=problem,
        exploration_solver=exploration_solver,
        exploitation_solver=exploitation_solver,
        max_decision_steps=max_decision_steps,
        search_steps_per_decision=search_steps_per_decision,
        max_search_steps=max_search_steps,
        reward_clip=reward_clip,
        log_type=log_type,
        log_dir=log_dir,
        session_id=session_id,
        emit_init_summary=emit_init_summary,
    )