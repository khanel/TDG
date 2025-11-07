# Final Report: A Dynamic, Multi-Objective Reward Framework

## 1. Reward Design Goals
- Balance solution quality with evaluation cost so the RL orchestrator adapts behaviour as the budget shrinks.
- Blend extrinsic objectives (fitness improvement) with intrinsic incentives (landscape understanding) to avoid sparse-reward traps ([Pathak et al., 2017](https://arxiv.org/abs/1708.07902)).
- Encode budget awareness so the agent re-weights exploration and exploitation on the fly, as in budgeted RL formulations ([Carrara et al., 2019](https://arxiv.org/abs/1903.01004)).

The result is a reward signal that mirrors expert search heuristics: explore broadly when time is abundant, then exploit aggressively as the deadline approaches.

## 2. Core Principles
- **Hybrid motivation**: combine extrinsic and intrinsic signals to accelerate learning in sparse or deceptive landscapes, a standard technique in intrinsic-motivation RL ([Pathak et al., 2017](https://arxiv.org/abs/1708.07902)).
- **Dynamic weighting**: treat the reward as a multi-objective combination and modulate the contribution of each term with the remaining budget, consistent with multi-objective RL practices ([Wikipedia: Multi-objective Reinforcement Learning](https://en.wikipedia.org/wiki/Multi-objective_reinforcement_learning)).
- **Risk awareness**: bias the policy towards safer choices as the budget runs out, aligning with risk-sensitive RL guidance ([Chow & Ghavamzadeh, 2018](https://arxiv.org/abs/1802.04364)).

## 3. Reward Components
The total reward at step `t` is
```
R_t = R_progress + R_efficiency + R_exploration + R_decision
```

### 3.1 Progress Reward (`R_progress`)
- **Role**: primary driver for improving the best-known tour.
- **Computation**: `R_progress = (prev_best - current_best) / initial_best`.
- **Behaviour**: positive values reward improvements; negative values penalise regressions.
- **Rationale**: normalised fitness deltas provide scale-invariant learning targets for evolutionary search quality ([Wikipedia: Fitness Function](https://en.wikipedia.org/wiki/Fitness_function)).

### 3.2 Efficiency Penalty (`R_efficiency`)
- **Role**: introduce a time pressure that discourages wasteful actions.
- **Computation**: `R_efficiency = -C`, with `C` a small constant per step.
- **Behaviour**: imposes a steady cost so shorter trajectories are preferable.
- **Rationale**: mirrors anytime-algorithm pressure to deliver good answers quickly ([Wikipedia: Anytime Algorithm](https://en.wikipedia.org/wiki/Anytime_algorithm)).

### 3.3 Intrinsic Exploration Reward (`R_exploration`)
- **Role**: incentivise discovering new regions of the landscape.
- **Computation**: `R_exploration = current_diversity - previous_diversity`, reusing the `population_concentration` feature (lower concentration ⇒ higher diversity).
- **Behaviour**: rewards actions that expand the search frontier even if no immediate fitness gain occurs.
- **Rationale**: aligns with novelty-driven heuristics that maintain behavioural diversity to escape local minima ([Wikipedia: Novelty Search](https://en.wikipedia.org/wiki/Novelty_search)).

### 3.4 Strategic Decision Bonus (`R_decision`)
- **Role**: deliver one-off signals for phase switches and termination quality.
- **Switch bonus**: `Bonus_switch = (1 - norm_best_at_switch) * (1 - concentration_at_switch)`.
- **Termination bonus**: `Bonus_terminate = (1 - final_norm_best)`; optional premature-stop penalty `Penalty_terminate = -(1 - budget_used_ratio)`.
- **Behaviour**: rewards switching when a strong launchpad has been built and incentivises finishing with a high-quality solution.
- **Rationale**: hyper-heuristic frameworks condition rewards on high-level policy decisions to learn phase scheduling ([Wikipedia: Hyper-heuristic](https://en.wikipedia.org/wiki/Hyper-heuristic)).

## 4. Budget-Aware Weighting
Let `B` denote `budget_remaining ∈ [0, 1]`. Modulate the component weights as:
- Exploration weight: `w_explore = B`.
- Quality weight: `w_quality = 1 - B`.
- Efficiency weight: constant or increasing as `B → 0` (e.g., `w_cost = C`).

Combined reward:
```
R_total = ( (1 - B) * R_progress ) + ( B * R_exploration ) - C + R_decision
```

- **Early budget (B ≈ 1)**: exploration dominates, pushing the agent to widen coverage and gather information.
- **Late budget (B ≈ 0)**: quality term dominates, focusing effort on tightening the incumbent solution.
- **Throughout**: the efficiency penalty and strategic bonuses keep the agent accountable for runtime and pivotal decisions.

## 5. Implementation Notes
- Calibrate `C` so that the time penalty nudges the policy without overwhelming legitimate improvements; anneal upward if runs frequently exhaust the budget without switching.
- Smooth diversity estimates (e.g., EWMA) to prevent the exploration term from oscillating on noisy population snapshots.
- Gate `R_decision` updates to the exact timestep where the agent issues a phase switch or termination action to avoid double-counting.
- Log each component to diagnose whether the agent is leveraging the intended trade-offs and to support future reward shaping adjustments ([Wikipedia: Reward Shaping](https://en.wikipedia.org/wiki/Fuzzy_control#Reward_shaping)).

## 6. Final Recommendation
Adopt this dynamic, multi-component reward function so the orchestrator learns policies that mirror expert heuristics: probe broadly while resources are plentiful, consolidate aggressively when the clock winds down, and make strategic switches and terminations that respect both quality and cost constraints.
