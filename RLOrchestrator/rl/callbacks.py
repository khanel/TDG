"""Shared callbacks for RL training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import logging
import time


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
        logger: Optional[logging.Logger] = None,
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
        progress_candidates = [
            int(np.round(self.total_timesteps * (i / 100.0)))
            for i in range(1, 101)
        ]
        progress_candidates = [p for p in progress_candidates if p > 0]
        progress_sorted = sorted(set(progress_candidates))
        if progress_sorted and progress_sorted[-1] != self.total_timesteps:
            progress_sorted.append(self.total_timesteps)
        self._progress_thresholds = progress_sorted or [self.total_timesteps]
        self._next_progress_idx = 0
        self._segment_returns: list[float] = []
        self._segment_episode_count = 0
        self._all_episode_returns: list[float] = []
        self._logger = logger
        self._start_time: float = 0.0

    def _init_callback(self) -> None:
        env = self.training_env
        if env is None:
            raise RuntimeError("Callback requires an environment")
        n_envs = env.num_envs
        self._episode_returns = np.zeros(n_envs, dtype=np.float64)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

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
                if ep_ret > self._best_reward:
                    self._best_reward = ep_ret
                self._segment_returns.append(ep_ret)
                self._segment_episode_count += 1
                self._all_episode_returns.append(ep_ret)

        # Suppress mid-run progress logs; advance thresholds silently
        while (
            self._next_progress_idx < len(self._progress_thresholds)
            and self.num_timesteps >= self._progress_thresholds[self._next_progress_idx]
        ):
            self._segment_returns.clear()
            self._segment_episode_count = 0
            self._next_progress_idx += 1

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
        if self._logger is not None:
            self._logger.debug(f"Saving new best checkpoint to {path} (reward={best_reward:.3f})")
        self.model.save(path)

    def _on_training_end(self) -> None:
        if self._logger is None:
            return
        # Final training summary across the entire run (single line)
        total_eps = len(self._all_episode_returns)
        duration = max(0.0, time.time() - (self._start_time or time.time()))
        if total_eps > 0:
            arr = np.asarray(self._all_episode_returns, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            mn = float(np.min(arr))
            mx = float(np.max(arr))
            self._logger.info(
                f"Summary: episodes={int(total_eps)}, return_mean={mean:.4f}, return_std={std:.4f}, "
                f"return_min={mn:.4f}, return_max={mx:.4f}, total_timesteps={int(self.num_timesteps)}, duration_sec={duration:.1f}"
            )
