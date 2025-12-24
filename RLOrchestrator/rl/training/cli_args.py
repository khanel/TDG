from __future__ import annotations

import argparse
from typing import Any


def add_training_core_args(
    parser: Any,
    *,
    total_timesteps_default: int,
) -> argparse.ArgumentParser:
    parser.add_argument("--total-timesteps", type=int, default=int(total_timesteps_default))
    parser.add_argument("--progress-bar", action="store_true", default=False)
    return parser


def add_budget_args(
    parser: Any,
    *,
    max_decisions_default: str,
    search_steps_per_decision_default: str,
) -> argparse.ArgumentParser:
    parser.add_argument("--max-decisions", type=str, default=str(max_decisions_default))
    parser.add_argument(
        "--search-steps-per-decision",
        type=str,
        default=str(search_steps_per_decision_default),
    )
    parser.add_argument(
        "--budget-ratio",
        type=str,
        default=None,
        help="Two-sided budget multiplier range, e.g. '0.5-2.0' (applies to both max-decisions and search-steps-per-decision).",
    )
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    return parser


def add_vec_env_args(
    parser: Any,
    *,
    num_envs_default: int,
    vec_env_default: str = "auto",
) -> argparse.ArgumentParser:
    parser.add_argument("--num-envs", type=int, default=int(num_envs_default))
    parser.add_argument(
        "--vec-env",
        choices=["auto", "dummy", "subproc"],
        default=str(vec_env_default),
    )
    return parser


def add_model_io_args(
    parser: Any,
    *,
    model_output_default: str,
) -> argparse.ArgumentParser:
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--model-output", type=str, default=str(model_output_default))
    return parser


def add_basic_ppo_args(
    parser: Any,
    *,
    learning_rate_default: float = 3e-4,
) -> argparse.ArgumentParser:
    parser.add_argument("--ppo-learning-rate", type=float, default=float(learning_rate_default))
    return parser


def add_full_ppo_args(
    parser: Any,
    *,
    learning_rate_default: float = 3e-4,
    n_steps_default: int = 2048,
    batch_size_default: int = 64,
    epochs_default: int = 10,
    ent_coef_default: float = 0.01,
    gamma_default: float = 0.99,
    gae_lambda_default: float = 0.95,
    device_default: str = "auto",
) -> argparse.ArgumentParser:
    parser.add_argument("--ppo-learning-rate", type=float, default=float(learning_rate_default))
    parser.add_argument("--ppo-n-steps", type=int, default=int(n_steps_default))
    parser.add_argument("--ppo-batch-size", type=int, default=int(batch_size_default))
    parser.add_argument("--ppo-epochs", type=int, default=int(epochs_default))
    parser.add_argument("--ppo-ent-coef", type=float, default=float(ent_coef_default))
    parser.add_argument("--ppo-gamma", type=float, default=float(gamma_default))
    parser.add_argument("--ppo-gae-lambda", type=float, default=float(gae_lambda_default))
    parser.add_argument("--device", type=str, default=str(device_default))
    return parser


def add_performance_args(parser: Any) -> argparse.ArgumentParser:
    """Performance controls.

    These flags exist mainly to avoid CPU oversubscription when using multiple
    environments (especially with SubprocVecEnv).
    """

    parser.add_argument("--fast", action="store_true", default=False)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument(
        "--subproc-start-method",
        type=str,
        default=None,
        choices=["spawn", "forkserver", "fork"],
    )
    return parser
