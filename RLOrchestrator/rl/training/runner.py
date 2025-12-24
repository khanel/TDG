from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


VecEnvType = Literal["dummy", "subproc"]
SingleEnvMode = Literal["raw", "dummy"]


def choose_vec_env_type(*, vec_env: str, num_envs: int) -> VecEnvType:
    vec = str(vec_env).lower().strip()
    n = max(1, int(num_envs))
    if vec == "auto":
        return "subproc" if n > 1 else "dummy"
    if vec in {"dummy", "subproc"}:
        return vec  # type: ignore[return-value]
    return "subproc" if n > 1 else "dummy"


def build_vec_env(
    env_fns: Sequence[Callable[[], Any]],
    *,
    num_envs: int,
    vec_env_type: VecEnvType,
    single_env_mode: SingleEnvMode = "raw",
):
    n = max(1, int(num_envs))
    if n == 1:
        if single_env_mode == "dummy":
            return DummyVecEnv([env_fns[0]])
        return env_fns[0]()

    vec_type = choose_vec_env_type(vec_env=str(vec_env_type), num_envs=n)
    if vec_type == "subproc":
        # On newer Python versions, 'fork'/'forkserver' can be less stable with
        # heavy numeric stacks; allow overriding via env var.
        start_method = os.environ.get("SB3_SUBPROC_START_METHOD", "spawn")
        return SubprocVecEnv(list(env_fns), start_method=start_method)
    return DummyVecEnv(list(env_fns))


def normalize_model_output_path(model_output: str) -> Path:
    path = Path(model_output).expanduser()
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    return path


def load_or_create_ppo(
    *,
    checkpoint_path: Optional[Path],
    env: Any,
    create_kwargs: Mapping[str, Any],
) -> Tuple[Any, bool]:
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path).expanduser()
        if ckpt.exists():
            return PPO.load(ckpt, env=env), False

    params = dict(create_kwargs)
    policy = params.pop("policy", "MlpPolicy")
    return PPO(policy, env, **params), True
