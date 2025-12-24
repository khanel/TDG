import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Core.problem import ProblemInterface, Solution
from Core.search_algorithm import SearchAlgorithm
from RLOrchestrator.core.orchestrator import OrchestratorEnv


class _BoundsChangingProblem(ProblemInterface):
    def __init__(self):
        self._upper = 1.0

    def evaluate(self, solution: Solution) -> float:
        # Minimization.
        solution.fitness = float(np.sum(np.asarray(solution.representation, dtype=float)))
        return float(solution.fitness)

    def get_initial_solution(self) -> Solution:
        sol = Solution([0, 0, 0], self)
        sol.evaluate()
        return sol

    def get_initial_population(self, size: int):
        return [self.get_initial_solution() for _ in range(max(1, int(size)))]

    def get_problem_info(self):
        return {"dimension": 3, "problem_type": "binary"}

    def get_bounds(self):
        return {"lower_bound": 0.0, "upper_bound": float(self._upper)}

    def regenerate_instance(self) -> bool:
        # Simulate domain randomization changing the fitness scale.
        self._upper = 100.0
        return True


class _NoOpSolver(SearchAlgorithm):
    def step(self):
        # No evolution needed for this test.
        return


def test_env_reset_refreshes_observation_bounds():
    problem = _BoundsChangingProblem()
    exploration = _NoOpSolver(problem, population_size=2)
    exploitation = _NoOpSolver(problem, population_size=2)
    exploration.initialize()
    exploitation.initialize()

    env = OrchestratorEnv(
        problem=problem,
        exploration_solver=exploration,
        exploitation_solver=exploitation,
        max_decision_steps=3,
        search_steps_per_decision=1,
    )

    # On reset, StageController calls regenerate_instance(); observation normalization
    # should refresh bounds accordingly.
    env.reset(seed=123)
    assert float(env.obs_comp.fitness_upper_bound) == 100.0
    env.close()


class _TaggedProblem(_BoundsChangingProblem):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = str(tag)


@pytest.mark.parametrize("tags", [("A", "B"), ("X", "Y")])
def test_env_can_swap_problem_per_episode(tags):
    tag_a, tag_b = tags

    def episode_factory(seed: int | None):
        # Alternate problems deterministically based on seed parity.
        if seed is not None and seed % 2 == 0:
            p = _TaggedProblem(tag_a)
        else:
            p = _TaggedProblem(tag_b)
        exp = _NoOpSolver(p, population_size=2)
        imp = _NoOpSolver(p, population_size=2)
        exp.initialize()
        imp.initialize()
        return p, exp, imp

    # Initialize with a dummy baseline; the factory should override on reset.
    base = _TaggedProblem("BASE")
    env = OrchestratorEnv(
        problem=base,
        exploration_solver=_NoOpSolver(base, population_size=2),
        exploitation_solver=_NoOpSolver(base, population_size=2),
        max_decision_steps=3,
        search_steps_per_decision=1,
        episode_factory=episode_factory,
    )

    env.reset(seed=2)
    assert getattr(env.problem, "tag", None) == tag_a

    env.reset(seed=3)
    assert getattr(env.problem, "tag", None) == tag_b
    env.close()
