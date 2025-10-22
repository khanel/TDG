"""Shared callbacks for RL training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PeriodicBestCheckpoint(BaseCallback):
    """Save checkpoints at fixed fractions of training when a new best episode reward is reached."""

    def __init__(
        self,
        total_timesteps: int,
        save_dir: Path,
        save_prefix: str,
        *,
        fraction: float = 0.05,
        verbose: int = 0,
        log_episodes: bool = True,
    ):
        super().__init__(verbose)
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if fraction <= 0 or fraction > 1:
            raise ValueError("fraction must be in (0, 1]")
        self.total_timesteps = int(total_timesteps)
        self.save_dir = Path(save_dir)
        self.save_prefix = save_prefix
        self.fraction = float(fraction)
        self._thresholds: list[int] = [
            int(np.round(self.total_timesteps * self.fraction * i))
            for i in range(1, int(np.floor(1 / self.fraction)) + 1)
        ]
        if self._thresholds[-1] != self.total_timesteps:
            self._thresholds.append(self.total_timesteps)
        self._next_threshold_idx = 0
        self._best_reward = -np.inf
        self._last_saved_best = -np.inf
        self._episode_returns: np.ndarray | None = None
        self._episode_count = 0
        self.log_episodes = bool(log_episodes)

    def _init_callback(self) -> None:
        env = self.training_env
        if env is None:
            raise RuntimeError("Callback requires an environment")
        n_envs = env.num_envs
        self._episode_returns = np.zeros(n_envs, dtype=np.float64)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        rewards: np.ndarray = np.asarray(self.locals.get("rewards"), dtype=np.float64)
        dones: np.ndarray = np.asarray(self.locals.get("dones"), dtype=bool)
        if self._episode_returns is None:
            raise RuntimeError("Callback not properly initialized")
        self._episode_returns += rewards

        for idx, done in enumerate(dones):
            if done:
                ep_ret = float(self._episode_returns[idx])
                self._episode_returns[idx] = 0.0
                self._episode_count += 1
                if self.log_episodes or self.verbose > 0:
                    print(f"Episode {self._episode_count}: reward={ep_ret:.3f} (env {idx})")
                if ep_ret > self._best_reward:
                    self._best_reward = ep_ret

        while (
            self._next_threshold_idx < len(self._thresholds)
            and self.num_timesteps >= self._thresholds[self._next_threshold_idx]
        ):
            if self._best_reward > self._last_saved_best:
                threshold = self._thresholds[self._next_threshold_idx]
                self._save_checkpoint(threshold, self._best_reward)
                self._last_saved_best = self._best_reward
            self._next_threshold_idx += 1

        return True

    def _save_checkpoint(self, threshold: int, best_reward: float) -> None:
        filename = f"{self.save_prefix}_best_step{threshold}.zip"
        path = self.save_dir / filename
        if self.verbose > 0:
            print(f"Saving new best checkpoint to {path} (reward={best_reward:.3f})")
        self.model.save(path)
