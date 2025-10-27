"""
Shared utilities for CLI parsing and environment configuration.
Merged from cli_utils.py and RL/environment_utils.py.
"""

from typing import Any, Callable, Dict, Tuple, Union
import argparse
import re

IntRangeSpec = Union[int, Tuple[int, int]]


def parse_int_range(spec: str, *, min_value: int, label: str) -> IntRangeSpec:
    """Parse a positive integer or inclusive range specification."""
    raw = (spec or "").strip()
    if not raw:
        raise ValueError(f"{label} cannot be empty")
    parts = [p.strip() for p in re.split(r"[,:-]", raw) if p.strip()]
    if not parts:
        raise ValueError(f"Invalid {label} value: {spec!r}")
    def _coerce(value: str) -> int:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must contain integers: {spec!r}") from exc
    if len(parts) == 1:
        val = max(min_value, _coerce(parts[0]))
        return val
    if len(parts) != 2:
        raise ValueError(f"Invalid {label} value: {spec!r}")
    lo, hi = _coerce(parts[0]), _coerce(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    lo = max(min_value, lo)
    hi = max(lo, hi)
    if lo == hi:
        return lo
    return (lo, hi)


def int_range_type(min_value: int, label: str) -> Callable[[str], IntRangeSpec]:
    def _parser(text: str) -> IntRangeSpec:
        try:
            return parse_int_range(text, min_value=min_value, label=label)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return _parser


def parse_float_range(spec: str, *, label: str) -> Tuple[float, float]:
    """Parse a float range specification."""
    raw = (spec or "").strip()
    if not raw:
        raise ValueError(f"{label} cannot be empty")
    parts = [p.strip() for p in re.split(r"[,:-]", raw) if p.strip()]
    if not parts:
        raise ValueError(f"Invalid {label} value: {spec!r}")
    def _coerce(value: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must contain floats: {spec!r}") from exc
    if len(parts) == 1:
        val = _coerce(parts[0])
        return (val, val)
    if len(parts) != 2:
        raise ValueError(f"Invalid {label} value: {spec!r}")
    lo, hi = _coerce(parts[0]), _coerce(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)



def max_steps_config_from_spec(spec: IntRangeSpec, *, randomize_scalar: bool = False, scalar_ratio: float = 0.6) -> Dict[str, Any]:
    """Build environment config entries for max episode steps."""
    if isinstance(spec, tuple):
        low, high = int(spec[0]), int(spec[1])
        if low > high:
            low, high = high, low
        low = max(1, low)
        high = max(low, high)
        if low == high:
            return {"randomize_timesteps": False, "max_steps": low}
        return {"randomize_timesteps": True, "max_steps_range": [low, high]}
    value = max(1, int(spec))
    if randomize_scalar and scalar_ratio > 0.0:
        low = max(1, int(value * float(scalar_ratio)))
        if low < value:
            return {"randomize_timesteps": True, "max_steps_range": [low, value]}
    return {"randomize_timesteps": False, "max_steps": value}


def internal_cfg_from_spec(spec: IntRangeSpec) -> Dict[str, Any]:
    if isinstance(spec, tuple):
        low, high = int(spec[0]), int(spec[1])
        if low > high:
            low, high = high, low
        low = max(1, low)
        high = max(low, high)
        if low == high:
            return {"randomize_internal_steps": False, "internal_steps": low}
        return {"randomize_internal_steps": True, "internal_steps_range": [low, high]}
    value = max(1, int(spec))
    return {"randomize_internal_steps": False, "internal_steps": value}


def range_upper(spec: IntRangeSpec) -> int:
    return int(spec[1]) if isinstance(spec, tuple) else int(spec)


def range_lower(spec: IntRangeSpec) -> int:
    return int(spec[0]) if isinstance(spec, tuple) else int(spec)