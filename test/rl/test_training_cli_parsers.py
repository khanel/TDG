import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _assert_has_option(parser, option: str) -> None:
    assert option in parser._option_string_actions, f"Missing option {option}"


def _assert_default(parser, option: str, expected) -> None:
    action = parser._option_string_actions.get(option)
    assert action is not None, f"Missing option {option}"
    assert action.default == expected, f"Default for {option} expected {expected!r}, got {action.default!r}"


@pytest.mark.parametrize(
    "module_path, expected_defaults",
    [
        (
            "RLOrchestrator.knapsack.rl.train",
            {
                "--total-timesteps": 100000,
                "--max-decisions": "200",
                "--search-steps-per-decision": "10",
                "--reward-clip": 1.0,
                "--ppo-learning-rate": 3e-4,
                "--num-envs": 1,
                "--vec-env": "auto",
                "--model-output": "ppo_knapsack",
            },
        ),
        (
            "RLOrchestrator.maxcut.rl.train",
            {
                "--total-timesteps": 100000,
                "--max-decisions": "200",
                "--search-steps-per-decision": "1",
                "--reward-clip": 1.0,
                "--ppo-learning-rate": 3e-4,
                "--num-envs": 1,
                "--vec-env": "auto",
                "--model-output": "ppo_maxcut",
            },
        ),
        (
            "RLOrchestrator.tsp.rl.train",
            {
                "--total-timesteps": 100000,
                "--max-decisions": "200",
                "--search-steps-per-decision": "1",
                "--reward-clip": 1.0,
                "--ppo-learning-rate": 3e-4,
                "--num-envs": 1,
                "--vec-env": "auto",
                "--model-output": "ppo_tsp",
            },
        ),
        (
            "RLOrchestrator.nkl.rl.train",
            {
                "--total-timesteps": 100000,
                "--max-decisions": "200",
                "--search-steps-per-decision": "10",
                "--reward-clip": 1.0,
                "--ppo-learning-rate": 3e-4,
                "--num-envs": 4,
                "--vec-env": "auto",
                "--model-output": "results/models/ppo_nkl_solveragnostic",
                "--device": "auto",
            },
        ),
        (
            "RLOrchestrator.rl.train_generalized",
            {
                "--total-timesteps": 250000,
                "--max-decisions": "200",
                "--search-steps-per-decision": "10",
                "--reward-clip": 1.0,
                "--ppo-learning-rate": 3e-4,
                "--num-envs": 4,
                "--vec-env": "auto",
                "--model-output": "results/models/ppo_generalized",
                "--device": "auto",
                "--problems": "all",
            },
        ),
    ],
)
def test_training_modules_expose_build_parser_and_defaults(module_path: str, expected_defaults: dict):
    mod = __import__(module_path, fromlist=["build_parser"])
    assert hasattr(mod, "build_parser"), f"{module_path} must define build_parser()"

    parser = mod.build_parser()
    for opt, expected in expected_defaults.items():
        _assert_has_option(parser, opt)
        _assert_default(parser, opt, expected)


@pytest.mark.parametrize(
    "module_path",
    [
        "RLOrchestrator.knapsack.rl.train",
        "RLOrchestrator.maxcut.rl.train",
        "RLOrchestrator.tsp.rl.train",
        "RLOrchestrator.nkl.rl.train",
        "RLOrchestrator.rl.train_generalized",
    ],
)
def test_training_cli_group_titles_are_consistent(module_path: str):
    mod = __import__(module_path, fromlist=["build_parser"])
    parser = mod.build_parser()
    titles = {g.title for g in getattr(parser, "_action_groups", [])}
    assert "Training" in titles
    assert "Environment" in titles
    assert "PPO" in titles
    assert "Model I/O" in titles
