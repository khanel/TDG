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

## Demo SUT (in-repo)

For a reproducible end-to-end target, see:

- examples/sbst-demo-java-maven

It is a tiny Maven project configured to produce JaCoCo XML at:

- `target/site/jacoco/jacoco.xml`

## Run a single orchestrator episode

This uses the registry-wired SBST solvers and invokes the Java pipeline during evaluation.

From repo root:

- `python3 -m RLOrchestrator.sbst.rl.run_episode \
	--project-root examples/sbst-demo-java-maven \
	--build-tool maven \
	--targets com.example.sbstdemo.BaseLogic com.example.sbstdemo.ChildLogic com.example.sbstdemo.GrandChildLogic \
	--max-decisions 10 \
	--search-steps-per-decision 1`

## Stage-10 validation (optional integration test)

The unit tests are lightweight by default. To run the opt-in end-to-end pipeline test
that invokes Maven/JaCoCo against the demo SUT:

- `SBST_RUN_INTEGRATION=1 python3 -m pytest -q test/sbst/test_demo_sut_pipeline_integration.py`

Design doc: `docs/tdg_sbst_inheritance_plan.md`
