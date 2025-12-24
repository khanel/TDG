from __future__ import annotations

import time
from pathlib import Path

from RLOrchestrator.sbst.cache.fingerprints import build_fingerprint_snapshot, mtime_first_then_hash_unchanged


def test_mtime_first_then_hash_allows_touch_without_content_change(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")

    snap1 = build_fingerprint_snapshot([f])

    # Touch (mtime changes) but keep content identical.
    time.sleep(0.01)  # ensure mtime changes on fast FS
    f.write_text("hello", encoding="utf-8")

    ok, snap2 = mtime_first_then_hash_unchanged(previous=snap1, paths=[f])
    assert ok is True
    assert snap2[str(f)].sha256 == snap1[str(f)].sha256


def test_mtime_first_then_hash_detects_content_change(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    snap1 = build_fingerprint_snapshot([f])

    time.sleep(0.01)
    f.write_text("bye", encoding="utf-8")

    ok, snap2 = mtime_first_then_hash_unchanged(previous=snap1, paths=[f])
    assert ok is False
    assert snap2[str(f)].sha256 != snap1[str(f)].sha256


def test_mtime_first_then_hash_detects_deleted_file(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    snap1 = build_fingerprint_snapshot([f])

    f.unlink()

    ok, _snap2 = mtime_first_then_hash_unchanged(previous=snap1, paths=[f])
    assert ok is False
