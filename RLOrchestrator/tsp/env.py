"""
Environment wiring for the Traveling Salesperson Problem (TSP).
"""

from typing import Optional
from ..core.env_factory import create_env
from ..core.orchestrator import OrchestratorEnv
from .adapter import TSPAdapter
from .solvers.map_elites import TSPMapElites
from .solvers.pso import TSPParticleSwarm

def build_tsp_env(
    num_cities: int = 20,
    grid_size: float = 100.0,
    max_decision_steps: int = 100,
    search_steps_per_decision: int = 1,
    max_search_steps: Optional[int] = None,
    reward_clip: float = 1.0,
    session_id: Optional[int] = None,
) -> OrchestratorEnv:
    """
    Builds a ready-to-use OrchestratorEnv for the TSP problem.

    Args:
        num_cities: The number of cities in the TSP instance.
        grid_size: The size of the grid on which cities are placed.
        max_decision_steps: The maximum number of decisions the RL agent can make.
        search_steps_per_decision: The number of solver steps to run per agent decision.
        max_search_steps: The absolute maximum number of solver steps for an episode.
        reward_clip: The range to clip rewards to.
        session_id: A unique ID for logging.

    Returns:
        A configured OrchestratorEnv instance for TSP.
    """
    # 1. Create the problem instance using the adapter
    problem = TSPAdapter(num_cities=num_cities, grid_size=grid_size)

    # 2. Instantiate the solvers
    exploration_solver = TSPMapElites(problem)
    exploitation_solver = TSPParticleSwarm(problem)

    # 3. Use the factory to create the environment
    env = create_env(
        problem=problem,
        exploration_solver=exploration_solver,
        exploitation_solver=exploitation_solver,
        max_decision_steps=max_decision_steps,
        search_steps_per_decision=search_steps_per_decision,
        max_search_steps=max_search_steps,
        reward_clip=reward_clip,
        log_type='train_tsp',
        session_id=session_id,
    )
    
    return env