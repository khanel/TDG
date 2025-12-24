from __future__ import annotations

import math
import os
import re
import sys
from typing import Any

from RLOrchestrator.core.utils import parse_float_range, parse_int_range


_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def apply_performance_settings(args: Any) -> None:
    """Apply runtime settings that affect training throughput.

    Notes:
    - BLAS/OpenMP thread env vars are inherited by SubprocVecEnv workers.
    - PyTorch threading is set in-process.
    """

    blas_threads = int(getattr(args, "blas_threads", 1) or 1)
    torch_threads = int(getattr(args, "torch_threads", 1) or 1)
    torch_interop_threads = int(getattr(args, "torch_interop_threads", 1) or 1)
    subproc_start_method = getattr(args, "subproc_start_method", None)

    for var in _BLAS_THREAD_ENV_VARS:
        os.environ[var] = str(max(1, blas_threads))

    if subproc_start_method:
        os.environ["SB3_SUBPROC_START_METHOD"] = str(subproc_start_method)

    try:
        import torch

        torch.set_num_threads(max(1, torch_threads))
        try:
            torch.set_num_interop_threads(max(1, torch_interop_threads))
        except Exception:
            # Can fail if set after parallel work has started.
            pass
    except Exception:
        # Torch might not be installed in some contexts (e.g. parser tests).
        pass


def apply_fast_preset(
    args: Any,
    *,
    num_envs_default: int,
    max_decisions_default: str,
    search_steps_per_decision_default: str,
    max_decisions_fast: str,
    search_steps_per_decision_fast: str,
) -> None:
    """Apply a speed-oriented preset.

    The preset only changes values that are still at their module defaults,
    so explicit CLI overrides win.
    """

    if not bool(getattr(args, "fast", False)):
        return

    # Prefer a stable start method when using SubprocVecEnv.
    if getattr(args, "subproc_start_method", None) is None:
        # forkserver is generally a good Linux default; fall back to spawn.
        start_method = "forkserver" if sys.platform != "win32" else "spawn"
        setattr(args, "subproc_start_method", start_method)

    # Increase env parallelism if user didn't set it.
    try:
        current_envs = int(getattr(args, "num_envs"))
    except Exception:
        current_envs = num_envs_default

    if current_envs == int(num_envs_default):
        cpu = os.cpu_count() or 8
        fast_envs = max(1, min(8, int(cpu)))
        setattr(args, "num_envs", fast_envs)

    # Reduce episode budgets if user didn't set them.
    if getattr(args, "max_decisions", None) == str(max_decisions_default):
        setattr(args, "max_decisions", str(max_decisions_fast))

    if getattr(args, "search_steps_per_decision", None) == str(search_steps_per_decision_default):
        setattr(args, "search_steps_per_decision", str(search_steps_per_decision_fast))


def apply_budget_ratio(args: Any) -> None:
    """Expand scalar budget values into a two-sided per-episode range.

    Example:
      --max-decisions 200 --search-steps-per-decision 10 --budget-ratio 0.5-2.0
    becomes:
      max_decisions='100-400', search_steps_per_decision='5-20'

    If a budget is already a range (e.g. '100-300'), it is left unchanged.
    """

    ratio_spec = getattr(args, "budget_ratio", None)
    if not ratio_spec:
        return

    lo_ratio, hi_ratio = parse_float_range(str(ratio_spec), label="budget-ratio")
    lo_ratio = float(lo_ratio)
    hi_ratio = float(hi_ratio)
    if lo_ratio > hi_ratio:
        lo_ratio, hi_ratio = hi_ratio, lo_ratio

    # Enforce the user's requirement: sample both below and above the base.
    if lo_ratio >= 1.0 or hi_ratio <= 1.0:
        raise ValueError("budget-ratio must span below and above 1.0 (e.g. '0.5-2.0')")

    def _expand(attr: str, *, label: str, min_value: int) -> None:
        raw = getattr(args, attr, None)
        if raw is None:
            return
        text = str(raw).strip()
        # If user already specified a range, keep it.
        if re.search(r"[,:-]", text):
            return

        base = parse_int_range(text, min_value=min_value, label=label)
        if isinstance(base, tuple):
            return

        low = int(math.floor(int(base) * lo_ratio))
        high = int(math.ceil(int(base) * hi_ratio))
        low = max(min_value, low)
        high = max(low, high)
        setattr(args, attr, f"{low}-{high}")

    _expand("max_decisions", label="max-decisions", min_value=1)
    _expand("search_steps_per_decision", label="search-steps-per-decision", min_value=1)
