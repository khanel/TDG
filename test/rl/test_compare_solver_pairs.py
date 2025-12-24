import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_compare_solver_pairs_module_exposes_build_parser():
    import RLOrchestrator.rl.compare_solver_pairs as mod

    assert hasattr(mod, "build_parser")
    parser = mod.build_parser()
    assert "--problem" in parser._option_string_actions
    assert "--episodes" in parser._option_string_actions
    assert "--num-cities" in parser._option_string_actions


def test_compare_solver_pairs_lists_tsp_pairs():
    import RLOrchestrator.rl.compare_solver_pairs as mod

    pairs = mod.list_solver_pairs("tsp")
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    # Pair entries are labeled by (explorer_class, exploiter_class)
    first = pairs[0]
    assert isinstance(first, tuple)
    assert len(first) == 2
    assert all(isinstance(x, str) and x for x in first)


def test_compare_solver_pairs_parses_fixed_num_cities_list():
    import RLOrchestrator.rl.compare_solver_pairs as mod

    assert mod.parse_num_cities_choices("20,50,200") == [20, 50, 200]
    assert mod.parse_num_cities_choices(" 20, 50 , 200 ") == [20, 50, 200]


def test_compare_solver_pairs_progress_every_is_reasonable():
    import RLOrchestrator.rl.compare_solver_pairs as mod

    assert mod._progress_every(1) == 0
    assert mod._progress_every(2) == 1
    assert mod._progress_every(9) == 1
    assert mod._progress_every(10) == 1
    assert mod._progress_every(100) == 10

