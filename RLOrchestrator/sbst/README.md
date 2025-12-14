# SBST TDG (Java)

This folder is the home of the **Search-Based Software Testing (SBST)** **Test Data Generation (TDG)** problem.

Scope (planned):
- Discover class inheritance hierarchies from Java source projects.
- Generate candidate tests (JUnit 5) from a Python-controlled search.
- Execute via Maven/Gradle and measure **branch coverage** (JaCoCo XML).
- Implement parent→child seeding along inheritance chains, with gating rules.

The actual SBST-TDG implementation will live in subpackages:
- `adapter.py`: `ProblemInterface` adapter (temporary scaffold)
- `discovery/`: Java project scanning + hierarchy discovery
- `generation/`: representation → JUnit rendering
- `execution/`: build tool invocation + test running
- `coverage/`: JaCoCo report parsing (branch coverage)
- `cache/`: mtime→hash caching for sources/build artifacts
- `pipeline/`: inheritance chain orchestration + seeding
- `solvers/`: minimal stub solvers so the problem can be registered end-to-end

Design doc: `docs/tdg_sbst_inheritance_plan.md`
