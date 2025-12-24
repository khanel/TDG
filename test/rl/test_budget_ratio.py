import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_apply_budget_ratio_expands_scalar_budgets_two_sided():
    from RLOrchestrator.rl.training.perf import apply_budget_ratio

    args = SimpleNamespace(
        budget_ratio="0.5-2.0",
        max_decisions="200",
        search_steps_per_decision="10",
    )
    apply_budget_ratio(args)

    assert args.max_decisions == "100-400"
    assert args.search_steps_per_decision == "5-20"


def test_apply_budget_ratio_does_not_override_existing_ranges():
    from RLOrchestrator.rl.training.perf import apply_budget_ratio

    args = SimpleNamespace(
        budget_ratio="0.5-2.0",
        max_decisions="100-300",
        search_steps_per_decision="5-15",
    )
    apply_budget_ratio(args)

    assert args.max_decisions == "100-300"
    assert args.search_steps_per_decision == "5-15"


def test_apply_budget_ratio_requires_two_sided_bounds():
    from RLOrchestrator.rl.training.perf import apply_budget_ratio

    args = SimpleNamespace(
        budget_ratio="1.0-2.0",
        max_decisions="200",
        search_steps_per_decision="10",
    )

    with pytest.raises(ValueError):
        apply_budget_ratio(args)
