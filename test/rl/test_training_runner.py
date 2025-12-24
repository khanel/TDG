import sys
from pathlib import Path
from unittest import mock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_choose_vec_env_auto_rules():
    from RLOrchestrator.rl.training.runner import choose_vec_env_type

    assert choose_vec_env_type(vec_env="auto", num_envs=1) == "dummy"
    assert choose_vec_env_type(vec_env="auto", num_envs=2) == "subproc"
    assert choose_vec_env_type(vec_env="dummy", num_envs=8) == "dummy"
    assert choose_vec_env_type(vec_env="subproc", num_envs=8) == "subproc"


def test_build_env_raw_single_uses_callable_result():
    from RLOrchestrator.rl.training.runner import build_vec_env

    class _Env:
        pass

    def make():
        return _Env()

    env = build_vec_env([make], num_envs=1, vec_env_type="dummy", single_env_mode="raw")
    assert isinstance(env, _Env)


def test_load_or_create_ppo_selects_checkpoint_and_reset_flag(tmp_path: Path):
    from RLOrchestrator.rl.training.runner import load_or_create_ppo

    dummy_env = object()

    ckpt = tmp_path / "m.zip"
    ckpt.write_text("x")

    fake_model = object()

    with mock.patch("RLOrchestrator.rl.training.runner.PPO") as PPO:
        PPO.load.return_value = fake_model
        model, reset_flag = load_or_create_ppo(
            checkpoint_path=ckpt,
            env=dummy_env,
            create_kwargs={"policy": "MlpPolicy"},
        )

        assert model is fake_model
        assert reset_flag is False
        PPO.load.assert_called_once()

    missing = tmp_path / "missing.zip"
    with mock.patch("RLOrchestrator.rl.training.runner.PPO") as PPO:
        PPO.load.side_effect = AssertionError("Should not load")
        PPO.return_value = fake_model
        model, reset_flag = load_or_create_ppo(
            checkpoint_path=missing,
            env=dummy_env,
            create_kwargs={"policy": "MlpPolicy", "verbose": 0},
        )
        assert model is fake_model
        assert reset_flag is True
        PPO.assert_called_once()


@pytest.mark.parametrize("suffix", [".zip", ".ckpt"])
def test_normalize_model_output_suffix(tmp_path: Path, suffix: str):
    from RLOrchestrator.rl.training.runner import normalize_model_output_path

    p = tmp_path / f"m{suffix}"
    out = normalize_model_output_path(str(p))
    assert out.suffix == ".zip"

    out2 = normalize_model_output_path(str(tmp_path / "no_suffix"))
    assert out2.suffix == ".zip"
