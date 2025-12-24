#!/usr/bin/env python3
"""Stage-8 smoke tests: SBST runs through the orchestrator wiring.

These tests intentionally avoid invoking real Java tooling by leaving `project_root=None`.
The SBST adapter/pipeline should still be callable and stable (always returns a fitness).
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RLOrchestrator.core.orchestrator import OrchestratorEnv
from RLOrchestrator.problems.registry import instantiate_problem


def _stage_map(stages):
    mapping = {binding.name: binding.solver for binding in stages}
    missing = {"exploration", "exploitation"} - mapping.keys()
    if missing:
        raise ValueError(f"Problem bundle missing stages: {sorted(missing)}")
    return mapping


@pytest.fixture
def sbst_bundle():
    return instantiate_problem(
        "sbst",
        adapter_kwargs={
            "seed": 42,
            "project_root": None,
            "targets": [],
            "timeout_seconds": 1,
        },
        solver_kwargs={
            "exploration": {"population_size": 6, "seed": 42},
            "exploitation": {"population_size": 6, "seed": 42},
        },
    )


def test_sbst_env_creation(sbst_bundle):
    stage_map = _stage_map(sbst_bundle.stages)

    env = OrchestratorEnv(
        problem=sbst_bundle.problem,
        exploration_solver=stage_map["exploration"],
        exploitation_solver=stage_map["exploitation"],
        max_decision_steps=5,
        search_steps_per_decision=1,
    )

    assert env.observation_space is not None
    assert env.action_space is not None
    env.close()


def test_sbst_env_reset_and_step(sbst_bundle):
    stage_map = _stage_map(sbst_bundle.stages)

    env = OrchestratorEnv(
        problem=sbst_bundle.problem,
        exploration_solver=stage_map["exploration"],
        exploitation_solver=stage_map["exploitation"],
        max_decision_steps=6,
        search_steps_per_decision=1,
    )

    obs, info = env.reset(seed=42)
    assert obs is not None
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(0)
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Ensure we have a best solution object with numeric fitness.
    best = env.get_best_solution()
    assert best is not None
    assert best.fitness is not None
    assert isinstance(float(best.fitness), float)

    env.close()
