# Meta-Orchestrator Design Manifesto (Minimal Two-Phase Controller)

Goal:
Design a problem-agnostic RL orchestrator that:
- Uses one exploration-only solver, one exploitation-only solver.
- Progresses through exactly two phases in one direction: Exploration → Exploitation → Termination.
- Learns only when to advance along this path vs. when to stay, to maximize efficiency across problems.

This manifesto encodes the MINIMAL control scheme as the default:
- No phase reversals.
- No arbitrary solver selection.
- No complex meta-operations.
- Only learn:
  - When to stop exploring and start exploiting.
  - When to stop exploiting and terminate.

It is structured as:
- Constraints: Non-negotiable properties of this minimal design.
- Design Axes: Explicit, limited choices you must make.
- Checkpoints: Concrete items to verify in code/experiments.

---

## 1. Core Concept and Scope (Minimal Model)

[Constraint 1.1] Two-level learning
- Outer level: RL policy (meta-controller).
- Inner level: Fixed, specialized black-box solvers.
  - One exploration solver.
  - One exploitation solver.
- Policy is trained over a distribution of tasks.

[Constraint 1.2] Strict role separation
- Exploration solver:
  - Configured purely for broad, diverse, global search.
- Exploitation solver:
  - Configured purely for local refinement around good candidates.
- Meta-policy:
  - Only controls transition timing and episode termination.
  - Never alters solver internals.

[Constraint 1.3] One-way phase progression
- Allowed global phase sequence:
  - Start: Exploration phase.
  - Then (optional, once): Exploitation phase.
  - Then (once): Termination.
- No transitions:
  - Exploitation → Exploration.
  - Post-termination continuation.

[Checkpoint 1.A]
- In [`RLOrchestrator/core/orchestrator.py`](RLOrchestrator/core/orchestrator.py:1):
  - Ensure global phase variable can only move:
    - exploration → exploitation → terminated.
- In each `RLOrchestrator/*/adapter.py`:
  - Designate exactly one exploration solver and one exploitation solver.
- Solvers:
  - Implement a shared interface; no embedded RL.

[Design Axis 1.D] Task distribution (still required)
- Choose and document:
  - (a) Single family (e.g., TSP variants).
  - (b) Mix of families (TSP, MaxCut, Knapsack, NKL).
- All must respect the same two-phase protocol.

---

## 2. Action Space: Minimal Two-Action, One-Way Control

[Constraint 2.1] Binary action set
At each decision step, the agent chooses exactly one of:

- 0 = STAY:
  - Remain in current phase.
  - If in Exploration: keep using exploration solver.
  - If in Exploitation: keep using exploitation solver.
- 1 = ADVANCE:
  - If in Exploration:
    - Irreversibly switch to Exploitation.
  - Else if in Exploitation:
    - Irreversibly Terminate (end episode).
  - Else if Terminated:
    - Invalid (episode already ended).

[Constraint 2.2] No other meta-actions
- No:
  - Restarts.
  - Phase toggling.
  - Arbitrary solver swapping.
- Only timing of the two transitions is learned.

[Constraint 2.3] Global semantics
- The meaning of actions is identical for all problems.
- No per-problem reinterpretation of 0/1.

[Checkpoint 2.A]
- In environment/orchestrator wiring:
  - Confirm:
    - Action 0: no phase change.
    - Action 1: exploration→exploitation if currently exploring; exploitation→termination if currently exploiting.
  - Assertions/logs to verify no illegal transitions occur.

---

## 3. Phase Process and Termination Semantics

[Constraint 3.1] Single exploration segment
- Start in exploration phase.
- Exploration continues while:
  - Agent chooses STAY (0), and budget not exhausted.
- When agent chooses ADVANCE (1) in exploration:
  - Switch to exploitation.
  - Cannot return to exploration.

[Constraint 3.2] Single exploitation segment
- In exploitation phase:
  - STAY (0): continue exploiting.
  - ADVANCE (1): terminate episode.
- Once terminated:
  - No further decisions.

[Constraint 3.3] Termination as part of ADVANCE
- There is no separate termination action:
  - The same ADVANCE action:
    - Means “commit” when in exploration (switch to exploit).
    - Means “stop” when in exploitation (terminate).
- Semantics:
  - The agent learns two stopping rules:
    - Stop exploring → start exploiting.
    - Stop exploiting → end.

[Checkpoint 3.A]
- Unit tests:
  - Simulate sequences of actions and confirm:
    - All valid sequences follow:
      - E (0/0/…) → E, until first 1 → X (0/0/…) → X, until next 1 → T.
    - No invalid backward or multiple exploit segments.

---

## 4. Observation Space: Support Two Stopping Decisions

Intent:
Provide just enough, but problem-agnostic, signal for two decisions:
- When is exploration “enough”?
- When is exploitation “enough”?

[Constraint 4.1] Problem-agnostic, bounded features
- All features normalized and shared across problems.
- No problem-specific semantics per index.

[Constraint 4.2] Minimal required signals
Observations must include (directly or via simple functions):

- budget_remaining:
  - Normalized [0,1].
- normalized_best_quality:
  - Normalized best objective with consistent direction and scale.
- improvement_signals:
  - Recent improvement rate / stagnation indicator.
- diversity_signals:
  - For exploration: is search still exploring broadly?
- phase_indicator:
  - 0 if Exploration, 1 if Exploitation.

[Constraint 4.3] No dependence on rich meta-actions
- Observation design should assume:
  - Only binary stay/advance decisions.
  - No need to encode controls that don’t exist (e.g., restart).

[Checkpoint 4.A]
- In [`RLOrchestrator/core/observation.py`](RLOrchestrator/core/observation.py:1):
  - Document:
    - Each feature’s formula and range.
    - How they are computed identically across problems.
  - Validate via logs:
    - Values lie within defined ranges.

[Design Axis 4.D]
- You may keep or drop complex landscape proxies.
- Constraint:
  - They must help answer:
    - “Continue exploring or switch?”
    - “Continue exploiting or terminate?”
  - Without breaking invariance.

---

## 5. Reward Design: Optimize Two Cutpoints

Intent:
Shape reward so that:
- The agent is rewarded for choosing good switch/stop times.
- The design is problem-agnostic, consistent, and simple.

[Constraint 5.1] Reward depends on normalized progress and budget use
- Core idea:
  - High final normalized_best_quality is good.
  - Using less budget for same quality is better.
- At each decision (or at termination), reward must:
  - Encourage:
    - Sufficient exploration to find good basins.
    - Timely switch to exploitation.
    - Timely termination when improvements saturate.

[Constraint 5.2] Simplicity over sophistication
- Avoid complex multi-term shaping that conflicts with minimal action space.
- Examples (to choose and tune explicitly, not all at once):

  - Dense incremental:
    - r_t = Δ(normalized_best_quality) - λ * normalized_cost_t

  - Or sparse terminal:
    - r_T = final_normalized_best_quality - λ * normalized_total_cost
    - r_t (intermediate) = small penalties or 0.

- Must be:
  - Same form across all problems.
  - Bounded and numerically stable.

[Constraint 5.3] No phase-specific hacks
- Do not hard-code:
  - “If TSP then different rule”.
- Exploration vs exploitation differences are reflected naturally through:
  - Their effects on progress/diversity, not conditionals on problem name.

[Checkpoint 5.A]
- For [`RLOrchestrator/core/reward.py`](RLOrchestrator/core/reward.py:1):
  - Ensure single, documented meta-objective.
  - Validate:
    - Similar reward distribution ranges across tasks.
    - Non-degenerate signals (not always negative).

---

## 6. Environment Mechanics

[Constraint 6.1] Decision steps
- Define a “decision step” as a chunk of inner iterations.
- At each decision step:
  - Agent observes state.
  - Chooses STAY or ADVANCE.
  - Environment:
    - Applies solver accordingly.
    - Updates phase if ADVANCE.

[Constraint 6.2] Termination conditions
- Episode ends when:
  - Budget exhausted (implicit termination).
  - Or exploitation phase receives ADVANCE (explicit termination).
- No hidden termination triggers beyond these and any clearly documented, problem-agnostic convergence rule.

[Constraint 6.3] Adapter contracts
- Adapters provide:
  - Fitness bounds.
  - Best-so-far.
  - Any diversity metrics used.
- All used in a problem-agnostic way.

[Checkpoint 6.A]
- Validate via rollouts:
  - Phase transitions and termination follow the minimal scheme.

---

## 7. Training and Evaluation under the Minimal Scheme

[Constraint 7.1] Train on distributions consistent with minimal control
- Use varied instances, but always:
  - E → (optional) X → T with 2 actions.

[Constraint 7.2] Evaluate what matters
- Metrics:
  - Final normalized_best_quality.
  - Budget-normalized performance.
  - Quality of:
    - Exploration duration.
    - Exploitation duration.
    - Termination timing.
- Baselines:
  - Fixed schedule (e.g., 50% explore, 50% exploit).
  - Always early/late switch heuristics.

[Checkpoint 7.A]
- Show that learned 2-action policy:
  - Outperforms simple fixed schedules consistently across problems.
  - Remains stable and interpretable.

---

## 8. Positioning and Extensions

[Constraint 8.1] Honest scope
- This minimal orchestrator is:
  - A learned schedule optimizer for a fixed two-stage solver pipeline.
  - Problem-agnostic within that modeling choice.
- Do not overclaim generality:
  - Full solver orchestration (restarts, multiple solvers, backtracking) is an extension.

[Design Axis 8.D] Future extensions (optional, not enabled now)
- If later needed and justified:
  - Add:
    - Restart.
    - Backward transitions.
    - More solver roles.
  - Only after:
    - Minimal scheme is validated and its limits are understood.

[Checkpoint 8.A]
- Implementation should keep:
  - Minimal scheme as clean baseline.
  - More complex variants guarded behind explicit flags/configs.

---

This manifesto encodes the “minimal” choice:
- Two phases, one direction.
- One exploration-only solver, one exploitation-only solver.
- Two actions: STAY vs ADVANCE.
- The RL agent’s sole responsibility:
  - Learn when to advance at each stage to maximize normalized solution quality per budget, across tasks.

All further design and implementation should be checked against this minimal contract unless you explicitly decide to move to a more expressive controller.