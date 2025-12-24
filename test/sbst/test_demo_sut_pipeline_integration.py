#!/usr/bin/env python3
"""Stage-10 validation: optional end-to-end pipeline run against the demo SUT.

This test is opt-in because it may invoke Maven/Java and can be slow.

Enable with:
- `SBST_RUN_INTEGRATION=1 python3 -m pytest -q`
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Core.problem import Solution
from RLOrchestrator.sbst.pipeline.models import SBSTConfig
from RLOrchestrator.sbst.pipeline.pipeline import SBSTPipeline


def _integration_enabled() -> bool:
    return str(os.getenv("SBST_RUN_INTEGRATION", "")).strip() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(not _integration_enabled(), reason="Set SBST_RUN_INTEGRATION=1 to enable")
def test_demo_sut_pipeline_one_eval_writes_coverage(tmp_path: Path):
    if shutil.which("mvn") is None:
        pytest.skip("mvn not found")

    demo_root = Path(__file__).parent.parent.parent / "examples" / "sbst-demo-java-maven"
    if not demo_root.exists():
        pytest.skip("demo SUT missing")

    # Ensure we don't leave generated tests behind in the SUT.
    before = set((demo_root / "src" / "test" / "java").rglob("GeneratedSBST_*.java"))

    cfg = SBSTConfig(
        seed=42,
        dimension=12,
        project_root=str(demo_root),
        build_tool="maven",
        targets=[
            "com.example.sbstdemo.BaseLogic",
            "com.example.sbstdemo.ChildLogic",
            "com.example.sbstdemo.GrandChildLogic",
        ],
        work_dir=str(tmp_path / "runs"),
        timeout_seconds=120,
        gating_enabled=True,
        gating_complete_threshold=0.0,  # force immediate completion to exercise gating transitions
        plateau_window=1,
        objective_granularity="method",
    )

    pipe = SBSTPipeline(cfg)

    sol = Solution({"genes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}, pipe)
    result = pipe.evaluate(sol)

    assert 0.0 <= float(result.fitness) <= 1.0
    if result.coverage_fraction is not None:
        assert 0.0 <= float(result.coverage_fraction) <= 1.0

    # Find the created run directory and validate coverage summary existence.
    runs_dir = Path(cfg.work_dir)
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    assert run_dirs, "expected at least one run dir"
    run_dir = sorted(run_dirs)[-1]

    coverage_summary = run_dir / "coverage" / "coverage_summary.json"
    assert coverage_summary.exists()

    # Generated tests should be cleaned up from the SUT after execution.
    after = set((demo_root / "src" / "test" / "java").rglob("GeneratedSBST_*.java"))
    assert after == before
