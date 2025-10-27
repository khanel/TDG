"""
Lightweight evaluation logger for recording agent behavior during evaluation.
Writes structured JSONL so downstream analysis is simple and language-agnostic.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_OBS_NAMES = [
    "budget_remaining",
    "normalized_best_fitness",
    "improvement_velocity",
    "stagnation_nonparametric",
    "population_concentration",
    "landscape_funnel_proxy",
    "landscape_deceptiveness_proxy",
    "active_phase",
]


@dataclass
class StepRecord:
    episode: int
    step: int
    phase_before: str
    action: int
    phase_after: str
    reward: float
    terminated: bool
    truncated: bool
    observation: List[float]
    best_fitness: Optional[float]
    improvement: Optional[float]
    decision_count: int
    search_steps_per_decision: int


@dataclass
class EpisodeSummary:
    episode: int
    total_steps: int
    total_return: float
    best_fitness: Optional[float]
    switch_steps: List[int]


class EvaluationLogger:
    def __init__(
        self,
        output_dir: Path,
        run_name: Optional[str] = None,
        *,
        obs_names: Optional[List[str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"eval_{ts}"
        self.obs_names = list(obs_names) if obs_names else DEFAULT_OBS_NAMES
        self.meta: Dict[str, Any] = extra_meta.copy() if isinstance(extra_meta, dict) else {}
        self.meta.update({
            "timestamp": ts,
            "obs_names": self.obs_names,
        })
        self.file_path = self.output_dir / f"{self.run_name}.jsonl"
        # Ensure file is created and meta header is written
        with self.file_path.open("w", encoding="utf-8") as f:
            header = {"event": "run_meta", "meta": self.meta}
            f.write(json.dumps(header) + "\n")

    def _write(self, obj: Dict[str, Any]) -> None:
        # Be defensive: ensure directory exists on every write (resilient to external cleanup)
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj) + "\n")
        except FileNotFoundError:
            # Retry once after ensuring directory
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj) + "\n")

    def log_episode_start(self, episode: int, meta: Optional[Dict[str, Any]] = None) -> None:
        rec = {"event": "episode_start", "episode": int(episode)}
        if meta:
            rec["meta"] = meta
        self._write(rec)

    def log_step(self, record: StepRecord) -> None:
        obj = {"event": "step"}
        obj.update(asdict(record))
        self._write(obj)

    def log_episode_end(self, summary: EpisodeSummary) -> None:
        obj = {"event": "episode_end"}
        obj.update(asdict(summary))
        self._write(obj)

    def path(self) -> Path:
        return self.file_path
