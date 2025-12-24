import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RLOrchestrator.core.env_factory import create_env


def test_prepare_run_artifacts_creates_dirs_and_config(tmp_path: Path):
    from RLOrchestrator.rl.training.run_artifacts import prepare_run_artifacts

    artifacts = prepare_run_artifacts(
        mode="train",
        problem="knapsack",
        model_output=str(tmp_path / "ppo_knapsack"),
        session_id=123,
        args={"total_timesteps": 100, "seed": 42},
    )

    assert artifacts.final_model_path.suffix == ".zip"
    assert artifacts.run_dir.name.endswith("_run123")
    assert artifacts.logs_dir.is_dir()
    assert artifacts.checkpoints_dir.is_dir()
    assert artifacts.config_path.is_file()

    payload = json.loads(artifacts.config_path.read_text())
    assert payload["mode"] == "train"
    assert payload["problem"] == "knapsack"
    assert payload["session_id"] == 123
    assert payload["final_model_path"].endswith("ppo_knapsack.zip")


def test_create_env_passes_episode_factory_through():
    # Reuse the small stubs from the env professionalization tests.
    from test.rl.test_training_env_professionalization import _TaggedProblem, _NoOpSolver

    def episode_factory(seed: int | None):
        tag = "A" if (seed is not None and seed % 2 == 0) else "B"
        p = _TaggedProblem(tag)
        exp = _NoOpSolver(p, population_size=2)
        imp = _NoOpSolver(p, population_size=2)
        exp.initialize()
        imp.initialize()
        return p, exp, imp

    base = _TaggedProblem("BASE")
    env = create_env(
        problem=base,
        exploration_solver=_NoOpSolver(base, population_size=2),
        exploitation_solver=_NoOpSolver(base, population_size=2),
        max_decision_steps=3,
        search_steps_per_decision=1,
        episode_factory=episode_factory,
        emit_init_summary=False,
    )

    env.reset(seed=2)
    assert getattr(env.problem, "tag", None) == "A"
    env.close()
