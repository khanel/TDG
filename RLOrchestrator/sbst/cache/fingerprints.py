from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    mtime_ns: int
    sha256: str

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_stat_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def build_fingerprint_snapshot(paths: Iterable[Path]) -> Dict[str, FileFingerprint]:
    snapshot: Dict[str, FileFingerprint] = {}
    for p in paths:
        try:
            if not p.exists() or not p.is_file():
                continue
            mtime = int(p.stat().st_mtime_ns)
            digest = sha256_file(p)
            snapshot[str(p)] = FileFingerprint(path=str(p), mtime_ns=mtime, sha256=digest)
        except OSError:
            continue
    return snapshot


def mtime_first_then_hash_unchanged(
    *,
    previous: Dict[str, FileFingerprint],
    paths: Iterable[Path],
) -> tuple[bool, Dict[str, FileFingerprint]]:
    """Return (unchanged, new_snapshot).

    Algorithm (required by master plan):
    1) Compare mtimes; if all equal -> unchanged.
    2) If any mtime differs -> compute sha256 and compare.
    3) If sha differs -> changed; else treat as unchanged and update mtime.

    Missing files are treated as a change (they simply won't appear in the new snapshot).
    """

    current_paths = [p for p in paths]

    # Quick mtime check.
    all_mtimes_same = True
    for p in current_paths:
        prev = previous.get(str(p))
        cur_mtime = safe_stat_mtime_ns(p)
        if prev is None or cur_mtime is None:
            all_mtimes_same = False
            break
        if int(prev.mtime_ns) != int(cur_mtime):
            all_mtimes_same = False
            break

    if all_mtimes_same and len(previous) > 0:
        # Still need to ensure no previously-known file disappeared.
        for prev_path in previous.keys():
            if not Path(prev_path).exists():
                return False, build_fingerprint_snapshot(current_paths)
        return True, previous

    # Hash compare for accuracy.
    new_snapshot: Dict[str, FileFingerprint] = {}
    changed = False

    # If a previous snapshot has paths not in current_paths, treat as change.
    current_set = {str(p) for p in current_paths}
    for prev_path in previous.keys():
        if prev_path not in current_set:
            changed = True
            break

    for p in current_paths:
        try:
            if not p.exists() or not p.is_file():
                changed = True
                continue
            mtime = int(p.stat().st_mtime_ns)
            digest = sha256_file(p)
            new_snapshot[str(p)] = FileFingerprint(path=str(p), mtime_ns=mtime, sha256=digest)

            prev = previous.get(str(p))
            if prev is None:
                changed = True
            else:
                if prev.sha256 != digest:
                    changed = True
        except OSError:
            changed = True

    return (not changed), new_snapshot
