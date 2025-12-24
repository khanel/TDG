from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

from .fingerprints import FileFingerprint


@dataclass(frozen=True)
class CacheEntry:
    key: str
    fingerprints: Dict[str, FileFingerprint]
    payload: Dict[str, Any]


def cache_dir(base_dir: Path) -> Path:
    return base_dir / "_cache"


def cache_path(base_dir: Path, *, namespace: str, key: str) -> Path:
    d = cache_dir(base_dir) / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def stable_key(parts: Dict[str, Any]) -> str:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def load_entry(base_dir: Path, *, namespace: str, key: str) -> Optional[CacheEntry]:
    path = cache_path(base_dir, namespace=namespace, key=key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None

    fps: Dict[str, FileFingerprint] = {}
    for k, v in (data.get("fingerprints") or {}).items():
        try:
            fps[k] = FileFingerprint(path=v["path"], mtime_ns=int(v["mtime_ns"]), sha256=str(v["sha256"]))
        except Exception:
            continue

    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    return CacheEntry(key=str(data.get("key") or key), fingerprints=fps, payload=payload)


def save_entry(base_dir: Path, *, namespace: str, key: str, fingerprints: Dict[str, FileFingerprint], payload: Dict[str, Any]) -> None:
    path = cache_path(base_dir, namespace=namespace, key=key)
    data = {
        "key": key,
        "fingerprints": {k: v.to_json_dict() for k, v in fingerprints.items()},
        "payload": payload,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
