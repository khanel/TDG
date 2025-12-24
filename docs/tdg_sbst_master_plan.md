# TDG SBST Master Plan (Java Branch Coverage)

Date: Dec 2025

This document is the **single, comprehensive plan** for integrating **SBST Test Data Generation (TDG)** for **Java branch coverage** into this repository.

It is intentionally long and explicit. It is meant to serve as:

- A shared reference for what SBST TDG means in this project.
- A record of what has already been done in the repo.
- A step-by-step roadmap of what we will implement next, in what order, and why.

---

## 0) The “least details” you still need to know

This section exists because SBST TDG has a lot of implicit assumptions. If two people implement “SBST TDG” without agreeing on these, they will build different systems that don’t compose.

### 0.1 Glossary (operational definitions used in this repo)

- **SUT (System Under Test)**: the Java project we are generating tests for.
- **Project root**: the directory that contains the build file (`pom.xml` or `build.gradle*`).
- **Target class**: a production class we want to cover. We prefer fully qualified names (FQNs) like `com.acme.Foo`.
- **Target method** (future): a method inside a target class; used if we move from class-level objectives to method-level objectives.
- **Candidate**: a single search individual. In SBST TDG, a candidate represents a **test suite** (one or more JUnit tests) in a deterministic encoding.
- **Test suite artifact**: the concrete Java source files written to disk (e.g., `GeneratedSBST_0123_00.java`).
- **Evaluation**: the expensive SBST loop for one candidate: generate → build/test → parse JaCoCo → compute fitness.
- **Fitness**: a scalar returned to the search algorithm. For SBST, fitness is derived from branch coverage (plus optional shaping later).
- **Branch coverage fraction**: $covered / (covered + missed)$ from JaCoCo branch counters.
- **Parent-first gating**: do not optimize child-only coverage until parent coverage reaches a completion condition.
- **Plateau heuristic**: detect no improvement over N evaluations and treat remaining target coverage as likely infeasible (for the current generator + configuration), then move on.

### 0.2 What we are explicitly NOT doing first (non-goals)

These are common SBST directions, but they add a lot of complexity. We postpone them until the baseline is stable.

- Deep mocking frameworks, bytecode instrumentation beyond JaCoCo, or custom runtime agents (other than what Maven/Gradle already do).
- Symbolic execution / concolic testing.
- Automatic Spring/Guice container wiring beyond whatever already happens when running unit tests.
- Full semantic feasibility proofs for branches.
- Mutation testing.

### 0.3 Prerequisites (what must exist on the machine)

The SBST pipeline is not “pure Python.” It relies on Java tooling.

- Java: a JDK installed and available as `java`/`javac`.
- Build tool:
  - Maven: `mvn` available, OR
  - Gradle: `gradle` available, OR
  - Gradle wrapper: `./gradlew` in the SUT (preferred when present).
- The SUT must be buildable in a clean environment.

Practical note:

- Some SUTs require environment variables, native libraries, services, credentials, or test data files. The SBST adapter must treat those as external requirements and fail gracefully when missing.

### 0.4 Artifacts and directory layout (so runs are debuggable)

SBST work produces a lot of artifacts. If we don’t standardize this early, debugging becomes impossible.

Proposed run directory structure (one run per orchestrator episode or per SBST evaluation batch):

- `runs/sbst/<timestamp>_<shortid>/`
  - `config.json` (fully resolved SBST config)
  - `discovery.json` (project discovery output)
  - `tests/` (generated Java test sources)
  - `build_logs/`
    - `stdout.txt`
    - `stderr.txt`
    - `exit_code.json`
    - `command.txt` (the exact Maven/Gradle invocation)
  - `coverage/`
    - `jacoco.xml` (or a copy/symlink to the located JaCoCo XML)
    - `coverage_summary.json`

Important: the system should be able to run without permanently modifying the SUT.

Two compatible strategies (choose one early, document it, and stick to it):

1. **Write tests into the SUT test tree** (easiest, but mutates the repo)
2. **Write tests into an isolated workspace** (preferred long-term): copy the SUT into a temp dir and run the build there.

MVP approach: start with (1) to get the pipeline working. Upgrade to (2) once the pipeline is stable.

### 0.5 Determinism rules (so results can be reproduced)

Determinism is critical because search is stochastic, build tools can be nondeterministic, and coverage can vary.

Rules:

- Every SBST evaluation must be reproducible from `(seed, candidate_representation, SBSTConfig)`.
- Generated filenames must be deterministic.
- Any random decisions inside discovery/generation must use a seeded RNG.

We cannot guarantee full determinism if the SUT is nondeterministic, but we can guarantee we do not add extra randomness.

### 0.6 Failure handling and scoring policy (do not crash; always return a fitness)

SBST evaluations fail frequently (compilation errors, runtime exceptions, failing tests). The adapter must always return a numeric fitness.

Proposed policy (explicitly implement and document):

- If generated tests do not compile → fitness = worst (e.g., `1.0` if minimizing, `0.0` if maximizing).
- If build execution fails before coverage report generation → fitness = worst.
- If tests run but JaCoCo XML is missing/unparseable → fitness = worst.
- Always write logs and preserve artifacts for debugging.

This keeps solvers and orchestrator stable even when candidates are “bad.”

### 0.7 Configuration surface (explicit knobs we will support)

The SBST adapter needs configuration that can be passed via the registry.

Minimum config keys (MVP):

- `project_root`: path to the SUT project root
- `build_tool`: `auto | maven | gradle`
- `targets`: list of class FQNs OR a discovery mode like `discover_all`
- `work_dir`: where SBST writes runs/artifacts
- `seed`: RNG seed

Extended config keys (still early but very likely required):

- `gradle_use_wrapper`: prefer `./gradlew` when available
- `maven_goals`: e.g., `test` then `jacoco:report`
- `gradle_tasks`: e.g., `test` then `jacocoTestReport`
- `jacoco_xml_path`: explicit path override (real projects vary)
- `timeout_seconds`: kill builds/tests that hang
- `max_tests_per_candidate`: keep candidates bounded
- `max_actions_per_test`: keep generated tests small
- `package_strategy`: how to choose the package for generated tests (match target package vs fixed)

### 0.8 Java access rules (cheat sheet for generation decisions)

Generated tests can only call what Java allows. This has direct consequences for where we place test classes and what we can invoke.

- `public`: callable from anywhere.
- `protected`:
  - callable from same package, OR
  - callable from a subclass in a different package.
- *package-private* (no modifier): callable only from the same package.
- `private`: not directly callable (ignore in MVP).

Implication for `package_strategy`:

- If we want to call package-private/protected members safely, we should generate tests **in the same package** as the target class.
- If the SUT uses strong encapsulation and we only generate tests in a fixed package, coverage potential may be much lower.

Practical MVP rule:

- Generate the test class in the target class’s package by default (one generated test class per target package), unless the user forces a fixed package.

### 0.9 Candidate representation (MVP encoding)

To integrate with the existing population-based metaheuristics, SBST needs a numeric-ish representation. The MVP encoding can be extremely simple while still deterministic.

MVP proposal (deterministic, bounded, easy to mutate):

- Candidate is a list of integers (genes).
- Interpret genes as a sequence of **actions**.
- Each action is a tuple encoded from consecutive genes:
  - select target class index
  - select method/constructor index
  - select argument values from a small domain

Constraints we enforce:

- Fixed maximum number of actions per candidate (bounded evaluation cost).
- Fixed maximum number of generated test methods per candidate.
- If an action maps to an invalid selection (index out of range), treat it as a no-op.

This encoding is intentionally crude. It exists to make the evaluation pipeline real first. We can evolve to richer encodings later.

### 0.10 Fitness direction and scaling (so solvers behave consistently)

The repository’s solvers generally assume a scalar objective where “lower is better” is common.

SBST objective (MVP):

- Let coverage fraction be $c \in [0, 1]$.
- Define fitness as $f = 1 - c$ (minimize).

Edge cases:

- If coverage cannot be measured (errors, missing report), return $f = 1$.
- If the target has no branches (`covered + missed == 0`), exclude it from gating and do not optimize it.

### 0.11 Inheritance processing details (what “internal hierarchy” means)

We need a precise definition so “process all internal hierarchies” is implementable.

- A class is **internal** if its source file is in the SUT production source roots.
- A parent is **external** if we cannot find a corresponding `.java` in the SUT production source roots.

Hierarchy rules:

- Build a directed graph `child -> parent` for internal classes.
- Stop traversal at the first external parent.
- Treat interfaces separately:
  - `implements` relationships matter for behavior, but the MVP parent-first gating is based on `extends` class inheritance.
  - We can add interface-aware gating later if needed.

Abstract classes:

- Abstract parent classes are still parents for gating purposes.
- We may not be able to instantiate abstract classes directly; generation must instead instantiate concrete children while targeting parent coverage.

### 0.12 Cache fingerprint scope (what files should invalidate results)

To make caching correct (not just fast), we must define what changes invalidate what.

Discovery invalidation inputs (MVP):

- Build files: `pom.xml`, `build.gradle`, `build.gradle.kts`, `settings.gradle*`, `gradle.properties`.
- Any `.java` under production source roots.

Coverage/execution invalidation inputs (MVP):

- Everything from discovery invalidation inputs, plus:
  - the generated test sources for the candidate
  - (optional) relevant resource files if they exist under `src/test/resources` and the project uses them

### 0.13 Safety note (running third-party code)

SBST TDG runs the SUT’s build and tests. If the SUT is untrusted, this can execute arbitrary code.

Policy:

- SBST TDG should be run only in a controlled environment (container/VM) for untrusted targets.
- The adapter should never exfiltrate secrets; avoid printing environment variables into logs by default.

---

## 1) What you need to know (project context)

### 1.1 What this repository is today
This repo is primarily a **reinforcement-learning guided meta-heuristic orchestrator**.

- The RL agent does not directly optimize problem solutions.
- Instead, it controls **phase transitions** between an **exploration** solver and an **exploitation** solver.
- The orchestrator enforces a unidirectional lifecycle (exploration → exploitation → termination) and transfers population state between solvers.

Important implication for SBST TDG:

- SBST becomes “just another problem” from the orchestrator’s perspective.
- But SBST’s evaluation is expensive and external (generate tests → execute build tool → parse coverage reports).
- Therefore, the SBST adapter must hide that complexity while producing a stable, measurable “fitness” signal.

### 1.2 What SBST TDG means here
**TDG = Test Data Generation** in the **SBST** sense:

- We are generating tests (primarily **JUnit 5** test cases) for a Java project.
- The objective is maximizing **branch coverage** of target code.
- Coverage will be measured using **JaCoCo** (default target), typically via JaCoCo XML reports.

In classic SBST terminology:

- A “candidate” is a test suite (or a test case) parameterized by some encoding.
- Fitness is derived from coverage achieved by executing the generated test suite.
- Search operators mutate/perturb candidates to improve coverage.

### 1.3 What “branch coverage” means operationally
We will treat “branch coverage” as:

- Coverage of decision outcomes recorded by JaCoCo.
- In JaCoCo XML, this appears via `<counter type="BRANCH" missed="…" covered="…"/>` at various levels (method/class/package).

Two practical realities:

1. Branch coverage depends on *executed control-flow*, which depends on inputs, environment, and code state.
2. Some branches can be **infeasible** (unreachable) without altering the system or using deep mocking.

We therefore need:

- A plateau heuristic to detect and handle “likely infeasible” branches.
- A gating strategy to keep optimization stable across inheritance hierarchies.

### 1.4 Java tooling assumptions
We will support Java projects that build with:

- Maven (`pom.xml`)
- Gradle (`build.gradle`, `build.gradle.kts`)

We will:

- Prefer running unit tests through the project’s own build tool.
- Generate tests as Java source under the project test source root:
  - Maven: `src/test/java/...`
  - Gradle: `src/test/java/...` (or configured alternative if detected)

Constraints and intent:

- “No manual Java coding” means the system generates tests programmatically; we are not writing bespoke tests by hand.
- But the generator itself will output Java code (JUnit) as artifacts.

---

## 2) Hard requirements and design constraints (captured so far)

These requirements are treated as **must-haves** unless explicitly relaxed.

### 2.1 Targets
- Input: a Java codebase (folders containing `.java`) and its build file.
- Objective: maximize branch coverage (JaCoCo).

### 2.2 Inheritance and access rules
We will handle inheritance in a way consistent with Java semantics:

- Process **internal class hierarchies** in full.
- Stop walking inheritance when a parent class is external (e.g., in a dependency jar).
- Generated tests may call **any accessible method** following Java access rules (public/protected/package-private depending on package and location).

### 2.3 Caching rules
Caching must follow:

- **mtime-first**: if file mtimes indicate no change, assume unchanged.
- If changed by mtime, then perform **hash compare**.
- On first run, store **baseline hashes**.

This is meant to avoid redoing expensive discovery/build/coverage parsing work.

### 2.4 Fitness gating across inheritance
The optimization objective across parent→child should be staged:

- Parent coverage objective runs first.
- Child-only objective activates only **after parent coverage is complete** (or reaches a defined “done” threshold).
- Infeasible branches are addressed via a plateau heuristic to avoid endless attempts.

---

## 3) What has already been done (repo state)

This section is about concrete, observable repository changes (not aspirations).

### 3.1 Restructuring completed
- Added SBST package scaffold under `RLOrchestrator/sbst/`.
- Moved algorithm implementations from root folders into `algorithms/`.
- Moved benchmark problems from root folders into `problems/`.
- Removed legacy scratch artifacts under `temp/` (and removed `legacy/` directory from disk previously).

### 3.2 SBST is registered in the orchestrator registry
SBST is now “instantiate-able” like the other problems:

- The registry has an `sbst` entry.
- It currently uses a placeholder surrogate objective in `SBSTAdapter` so solvers can run end-to-end without Java tooling.

This was done intentionally as a scaffold: the orchestrator integration is validated before the Java pipeline is implemented.

### 3.3 Documentation updated/added
- Restructure notes exist in `docs/repo_restructure_sbst_tdg.md`.

---

## 4) Where SBST TDG fits in the architecture

### 4.1 Current contracts in the repository
The repo already contains conceptual design docs:

- `docs/adapter_contract_design.md`
- `docs/solver_contract_design.md`

The codebase also contains a practical “registry-first” instantiation path via:

- `RLOrchestrator/problems/registry.py`

**Important:** SBST will primarily integrate via the registry entry, not via ad-hoc scripts.

### 4.2 SBST-specific adapter responsibilities
A real SBST adapter must do all of the following while still looking like a normal `ProblemInterface` to solvers:

1. **Candidate ↔ test mapping**
   - Convert candidate representation into a test suite artifact (JUnit source files).

2. **Project execution**
   - Run Maven/Gradle tests and trigger JaCoCo report generation.

3. **Coverage parsing and fitness computation**
   - Parse JaCoCo XML.
   - Compute fitness = something like `1 - branchCoverage` (or a richer shaping signal).

4. **Caching**
   - Avoid re-running steps when irrelevant.

5. **Artifact management**
   - Store generated tests, logs, and reports in a reproducible structure.

### 4.3 SBST pipeline (conceptual)
At evaluation time, the pipeline is:

1. Discover project config (build tool, source roots, test roots, targets)
2. Generate tests from candidate
3. Write tests into a temp or dedicated test directory
4. Execute tests via build tool
5. Produce JaCoCo XML
6. Parse coverage and compute fitness
7. Return fitness to the search algorithm/orchestrator

---

## 5) Proposed SBST data model (what we will implement)

This is not code yet; it is the “shape” we want so implementation stays coherent.

### 5.1 Project-level model
- `ProjectUnderTest`
  - `root_path`
  - `build_tool` (maven|gradle)
  - `source_roots`, `test_roots`
  - `target_classes` (explicit list or discovered)
  - `classpath` / build output locations (if needed)

### 5.2 Target hierarchy model
- `TypeId` (package + class name)
- `HierarchyNode`
  - `type_id`
  - `parent_type_id` (optional)
  - `is_external_parent` (bool)
  - `children` (list)

We need this because SBST gating uses parent completion before child targeting.

### 5.3 Coverage model
- `CoverageSummary`
  - `branches_covered`, `branches_missed`
  - coverage fraction `$covered / (covered + missed)$`
  - optionally method/class/package breakdown

- `CoverageReport`
  - overall summary
  - by-class summaries
  - (optional future) by-method and branch identifiers

### 5.4 Caching model
- `FileFingerprint`
  - `path`
  - `mtime`
  - `sha256`

- `CacheEntry`
  - “discovery” results (project layout, targets)
  - “coverage parse” results (parsed summary)
  - “test execution” results (stdout/stderr, exit code)

---

## 6) Implementation plan (detailed, comprehensive, in-order)

This plan is intentionally staged so we can demonstrate correctness early.

### Stage 0 — Stabilize after restructure (sanity)
Goal: confirm the repo is in a healthy baseline state before we add SBST complexity.

Work:

- Run Python compilation sanity check across the repo.
- Run minimal import checks for `RLOrchestrator/problems/registry.py` and `RLOrchestrator/sbst/`.
- Fix only issues caused by restructuring, not unrelated pre-existing issues.

Acceptance criteria:

- `python -m compileall` succeeds (or failures are understood and unrelated).
- Importing the registry and instantiating a known problem works.

### Stage 1 — Define the real SBST adapter contract (what “a step” means)
Goal: turn `SBSTAdapter` from a surrogate into a contract that matches SBST’s reality.

Decisions to make explicit in code:

- What is a candidate representation?
  - Minimal initial encoding should allow deterministic reproduction.
  - It must map to concrete JUnit tests.

- What does `evaluate()` do?
  - It must execute the full SBST pipeline (generate → run → parse).
  - It must be robust and return a fitness even on failures.

- What does the adapter expose as “problem info” and “bounds”?
  - For SBST, “bounds” may not be numeric; we may emulate numeric genes for solvers.

Deliverables:

- Update `RLOrchestrator/sbst/adapter.py` to:
  - accept config: project root, build tool override, target selection
  - define directories for artifacts
  - delegate evaluation to pipeline components in `RLOrchestrator/sbst/...`

Acceptance criteria:

- SBST adapter can be instantiated by registry with config overrides.
- Evaluation returns a stable numeric fitness and records artifacts.

### Stage 2 — Project discovery & build detection
Goal: given a filesystem path, produce a `ProjectUnderTest`.

Work details:

- Detect Maven vs Gradle.
  - Maven if `pom.xml` exists at/above root.
  - Gradle if `build.gradle` or `build.gradle.kts` exists.
  - If both exist (multi-build projects), require an explicit override to avoid ambiguous behavior.
  - Prefer Gradle wrapper (`./gradlew`) when present because it pins the Gradle version.

- Determine source/test roots.
  - Default conventions first.
  - Later: optionally parse build files for overrides.
  - Always record which roots were assumed vs discovered so runs remain explainable.

- Identify target classes.
  - Options (start simple):
    - user-provided list of fully qualified class names
    - discover all production classes under source roots
  - Discovery rule (MVP): treat any `.java` under `src/main/java` (or detected production roots) as a production class.
  - Exclude known build output roots: `target/`, `build/`, `out/`, `.gradle/`.

- Discover internal inheritance hierarchies.
  - Parse `.java` using a lightweight approach first (regex/heuristic), then upgrade.
  - Stop at external parents (not in project sources).
  - Minimum extraction required:
    - `package` statement
    - `class`/`interface`/`enum` name
    - `extends <Parent>`
  - Treat unresolved parents as external.

Acceptance criteria:

- For a given Java project root, we can produce:
  - build tool
  - source/test roots
  - a list of classes and parent relationships

### Stage 3 — Execution and JaCoCo coverage collection
Goal: reliably run the project tests and obtain JaCoCo XML.

Work details:

- Maven path:
  - Ensure JaCoCo plugin is configured to produce XML.
  - If not configured, decide strategy:
    - minimal non-invasive: run with a preconfigured JaCoCo argLine if possible
    - or generate temporary build config overlays
  - Baseline commands (subject to project override):
    - `mvn -q test`
    - `mvn -q jacoco:report`
  - Typical JaCoCo XML location (not guaranteed): `target/site/jacoco/jacoco.xml`

- Gradle path:
  - Ensure `jacocoTestReport` runs and produces XML.
  - Baseline commands (subject to project override):
    - `./gradlew test jacocoTestReport` (preferred)
    - `gradle test jacocoTestReport` (fallback)
  - Typical JaCoCo XML location (not guaranteed): `build/reports/jacoco/test/jacocoTestReport.xml`

- Capture stdout/stderr and exit codes.
- Ensure the adapter can run “tests only” without other lifecycle steps when possible.
 - Add timeouts and hard-kill support so search cannot hang indefinitely.

Acceptance criteria:

- Running the pipeline produces a JaCoCo XML report at a known location.

### Stage 4 — JaCoCo XML parsing into a stable coverage model
Goal: parse JaCoCo output and compute branch coverage.

Work details:

- Parse overall and per-class `<counter type="BRANCH" ...>`.
- Provide both:
  - raw counts
  - normalized coverage fraction

Parsing corner cases we must define up-front:

- If `covered + missed == 0` for a class (no branches exist), treat it as “not a meaningful branch target” and exclude it from parent/child gating objectives.
- Always normalize to a float in `[0, 1]`.
- If the XML is missing or malformed, treat coverage as unavailable and fall back to the failure fitness policy.

- Decide what “completion” means for gating.
  - simplest: parent is “complete” if coverage fraction == 1.0
  - pragmatic alternative: treat >= 0.99 as complete due to measurement quirks

Acceptance criteria:

- Given a JaCoCo XML file, we can compute branch coverage reliably.

### Stage 5 — Caching (mtime-first then hash)
Goal: avoid unnecessary work.

Caching layers:

- Discovery cache:
  - invalidates when build files or Java sources change.

- Execution/coverage cache:
  - invalidates when generated tests change OR production sources change.

Algorithm (required):

1. Read mtimes for relevant files.
2. If mtimes unchanged, reuse cached results.
3. If mtimes changed, compute sha256 and compare to cached baseline.
4. On first run, store baseline hashes.

Additional cache-key details (needed so the cache is correct):

- The cache key must include:
  - SBST config (at least the parts that affect evaluation)
  - the candidate representation (or a stable digest of it)
  - relevant SUT file fingerprints (sources + build files)
- If any of these change, cached coverage must not be reused.

Acceptance criteria:

- Re-running SBST with no changes avoids full re-execution.
- Changing a single source file invalidates appropriately.

### Stage 6 — Test generation MVP (JUnit 5) with “no manual Java coding”
Goal: generate *compilable* tests consistently.

MVP strategy (start conservative):

- Generate tests that:
  - instantiate classes using accessible constructors
  - call accessible methods with simple generated inputs
  - include assertions that avoid flakiness (or minimal assertions)

- Handle method argument generation:
  - primitives + common standard library types
  - nullability considerations

- Use deterministic generation seeded from candidate representation.

Concrete MVP constraints (write them into code as validation so generation cannot silently explode):

- Keep test suites small:
  - limit number of generated test methods per class
  - limit number of calls (“actions”) per test method
- Use stable naming:
  - classes: `GeneratedSBST_<candidate_id>_<index>.java`
  - methods: `test_<seed>_<index>()`
- Avoid flaky assertions:
  - MVP uses either no assertions or only structural assertions (e.g., “does not throw”).

Method selection rules (MVP):

- Prefer public instance methods.
- Include static methods if accessible.
- Skip methods that accept complex types we cannot construct yet.
- If no callable methods exist, still generate a trivial compile-pass test so the evaluation can proceed and be scored.

Acceptance criteria:

- Generated tests compile and run through Maven/Gradle.
- Even if tests fail sometimes, the system records failure and yields a fitness signal.

### Stage 7 — Fitness shaping and inheritance gating
Goal: implement the parent-first objective switching.

Work details:

- Define targets in an order driven by hierarchy:
  - parents before children

- Fitness function logic:
  - If parent not complete:
    - fitness targets parent branch coverage only
  - Else:
    - fitness targets child-only branch coverage

- Infeasible branch plateau heuristic:
  - Track lack of improvement over N evaluations.
  - If plateau persists, mark remaining branches as likely infeasible for the current target and move on.

Minimum plateau specification (so it is not vague):

- Track best-achieved coverage for the current target objective.
- If no improvement for `plateau_window` evaluations (e.g., 30–100), then:
  - mark target as “plateaued”
  - allow progression to next target (or next phase) based on policy

This is not claiming true infeasibility; it is a practical termination/escape condition.

Acceptance criteria:

- The system does not begin “child-only” optimization until parent is marked complete.
- Plateaus do not cause endless loops.

### Stage 8 — Search loop integration (explorer/exploiter)
Goal: make SBST fully “first-class” in the orchestrator.

Work details:

- Provide baseline explorer/exploiter implementations for SBST.
- Ensure candidate representations flow correctly through solvers.
- Ensure `OrchestratorEnv` can train/evaluate with SBST like other problems.

Acceptance criteria:

- We can run an SBST episode end-to-end using the orchestrator and registry.

### Stage 9 — End-to-end demonstration target
Goal: a reproducible demo.

Options:

- Add a tiny example Java project in-repo (preferred for reproducibility), OR
- Document how to point SBST at an external Java project.

Acceptance criteria:

- A documented command generates tests and reports branch coverage.

### Stage 10 — Testing and validation
Goal: avoid regressions and hard-to-debug failures.

Test strategy:

- Unit tests for:
  - build detection
  - cache invalidation rules
  - JaCoCo XML parsing
  - gating logic

- Avoid heavy integration tests until the pipeline is stable.

---

## 7) Risks and mitigation

### 7.1 Build tool diversity
Risk: Maven/Gradle projects are configured in many ways.

Mitigation:

- Start with convention-over-configuration.
- Provide explicit overrides in SBST config (paths, tasks/goals).
- Keep logs and artifacts for debugging.

### 7.2 Test flakiness and nondeterminism
Risk: generated tests can be flaky or dependent on environment.

Mitigation:

- Start with conservative, deterministic generation.
- Prefer pure methods and simple inputs first.
- Record execution logs and isolate work directories.

### 7.3 Infeasible branches
Risk: branch coverage can be limited without mocks or environmental control.

Mitigation:

- Plateau heuristic.
- Track “likely infeasible” per target.

### 7.4 Performance
Risk: executing Maven/Gradle repeatedly is slow.

Mitigation:

- Aggressive caching.
- Keep test suites minimal early.
- Later: incremental compilation or test selection.

---

## 8) Concrete next steps (what we do immediately)

1. Run Stage 0 sanity checks.
2. Replace the surrogate objective in `SBSTAdapter` with a pipeline-backed structure (still MVP) and define config surface.
3. Implement discovery + minimal build detection for Maven/Gradle.
4. Implement JaCoCo XML parsing for branch counters.

These steps create an end-to-end “thin slice”: even if test generation is simple, we can already measure and optimize something real.

---

## 9) Appendix: Directory map (where code should go)

- `RLOrchestrator/sbst/adapter.py`
  - SBSTAdapter, SBSTConfig, wiring into pipeline

- `RLOrchestrator/sbst/discovery/`
  - build tool detection, source/test roots, target discovery, hierarchy extraction

- `RLOrchestrator/sbst/generation/`
  - candidate → JUnit generation (JUnit 5)

- `RLOrchestrator/sbst/execution/`
  - run Maven/Gradle, capture output, produce reports

- `RLOrchestrator/sbst/coverage/`
  - JaCoCo XML parsing, coverage summaries

- `RLOrchestrator/sbst/cache/`
  - mtime-first then hash cache

- `docs/repo_restructure_sbst_tdg.md`
  - structural notes and how SBST was introduced into the repo
