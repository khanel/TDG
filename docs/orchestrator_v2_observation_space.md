# Final Report and Definitive Proposal: The Orchestrator-V2 Observation Space

## 1. The Philosophy of the Ultimate Observation Space
- **Sufficiency**: capture enough state to separate productive exploration from stagnation and to surface when different control actions are warranted.
- **Efficiency**: keep every feature inexpensive so that observation costs remain a negligible fraction of a single solver evaluation.
- **Controllability**: favour prescriptive signals that help the agent predict the impact of its options (continue, switch, terminate) instead of passively reacting to lagging indicators.

This document redefines the observation space so that the RL orchestrator receives a richer, predictive view of the TSP search dynamics with minimal computational overhead.

## 2. The Orchestrator-V2 Observation Space (8D)
1. `budget_remaining`
2. `normalized_best_fitness`
3. `improvement_velocity`
4. `stagnation_nonparametric`
5. `population_concentration`
6. `landscape_funnel_proxy`
7. `landscape_deceptiveness_proxy`
8. `active_phase`

## 3. Detailed Feature Breakdown

### Feature 1: `budget_remaining`
- **Concept**: normalized countdown of the remaining evaluation budget.
- **Metric**: `(max_steps - current_step) / max_steps`.
- **Why it matters**: supplies time-horizon awareness so the agent scales exploration vs. exploitation pressure appropriately.
- **Research note**: Anytime algorithm control routinely uses normalized time or resource budgets as primary state ([Anytime Algorithms](https://en.wikipedia.org/wiki/Anytime_algorithm)).

### Feature 2: `normalized_best_fitness`
- **Concept**: progress gauge relative to known instance bounds.
- **Metric**: `(best_fitness - lower_bound) / (upper_bound - lower_bound)`.
- **Why it matters**: anchors the overall notion of success and keeps the agent grounded in absolute progress.
- **Research note**: Foundational optimisation literature treats fitness normalisation as the standard lens for cross-instance comparison ([Fitness Function](https://en.wikipedia.org/wiki/Fitness_function)).

### Feature 3: `improvement_velocity`
- **Concept**: smoothed rate of improvement on the current best solution.
- **Metric**: EWMA of successive deltas in `normalized_best_fitness`, `v_t = α·Δ_t + (1-α)·v_{t-1}`.
- **Why it matters**: distinguishes rapid gains from slow grind, allowing proactive phase switches when velocity decays.
- **Research note**: Exponentially weighted moving averages are a classic tool for on-line trend detection in noisy signals ([Exponential Moving Average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average)).

### Feature 4: `stagnation_nonparametric`
- **Concept**: statistical test for meaningful progress in recent history.
- **Metric**: `1 - p_value` from a Mann–Whitney U test comparing two adjacent windows of best fitness values.
- **Why it matters**: replaces brittle stagnation counters with a robust, noise-tolerant detector of distributional change.
- **Research note**: Non-parametric change detection via Mann–Whitney U is lightweight and resilient to non-Gaussian noise ([Mann–Whitney U Test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)).

### Feature 5: `population_concentration`
- **Concept**: direct measure of population spread around its centroid.
- **Metric**: mean Euclidean distance of normalized individuals from the population centroid, scaled by `sqrt(d)` for dimension `d`.
- **Why it matters**: captures the natural collapse from exploration to exploitation and flags diversity loss earlier than entropy proxies.
- **Research note**: Diversity-aware management is a central theme in evolutionary landscape analysis ([Evolutionary Landscape and Diversity Management](https://arxiv.org/abs/1510.07163)).

### Feature 6: `landscape_funnel_proxy`
- **Concept**: probe for smooth basin structure around the incumbent best solution.
- **Metric**: Spearman correlation between sampled neighbour distances from `S_best` and their fitness values.
- **Why it matters**: a strong positive correlation evidences funnel-shaped landscapes where exploitation should pay off.
- **Research note**: Fitness-distance correlation is a core diagnostic in fitness landscape analysis ([Landscape Analysis Review](https://arxiv.org/abs/1406.0194)).

### Feature 7: `landscape_deceptiveness_proxy`
- **Concept**: single long-jump evaluation to sense alternative basins.
- **Metric**: normalized fitness of a heavily mutated solution `S_far` relative to `S_best`.
- **Why it matters**: identifies deceptive landscapes where distant regions outperform the current basin, prompting renewed exploration.
- **Research note**: Quality-diversity methods such as MAP-Elites explicitly sample distant niches to overcome deception ([MAP-Elites](https://arxiv.org/abs/1504.04909)).

### Feature 8: `active_phase`
- **Concept**: categorical encoding of the currently active solver or phase.
- **Metric**: one-hot vector (e.g., exploration vs. exploitation).
- **Why it matters**: contextualises all other signals—tight concentration in exploration implies failure; tight concentration in exploitation signals convergence.
- **Research note**: Hyper-heuristic orchestration relies on knowing which heuristic is engaged to learn switching policies ([Hyper-heuristics](https://en.wikipedia.org/wiki/Hyper-heuristic)).

## 4. Driving the Desired Trajectory
- **Early exploration**: High `budget_remaining` and `population_concentration`, spiky `improvement_velocity`, occasional wins in `landscape_deceptiveness_proxy`. Agent keeps exploration active.
- **Transition phase**: `improvement_velocity` decays, `stagnation_nonparametric` drops, `population_concentration` naturally shrinks, `landscape_funnel_proxy` turns positive. Agent switches to exploitation.
- **Late exploitation**: Low `budget_remaining`, `active_phase` signals exploitation, `population_concentration` collapses, `improvement_velocity` and `stagnation_nonparametric` trend to zero. Agent terminates confidently.

## 5. Final Recommendation
Replace the current 7-feature observation space with the above 8-feature Orchestrator-V2 design. The upgrade shifts the controller from reactive monitoring to predictive guidance, enabling smarter phase scheduling and termination decisions while maintaining negligible computational overhead.
