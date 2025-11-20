# Observation Space Baseline & Experiment Plan

## Purpose
- Consolidate every observation-space insight scattered across `observation_space_design.md`, `ela_driven_observation_space.md`, `Proposed Observation Features.md`, `observation_feature_candidates.md`, and `tmp.md`.
- Freeze a **minimal-yet-sufficient 3D baseline** so training scripts, adapters, and reward work all target the same schema.
- Maintain a prioritized backlog of candidate signals with clear cost/benefit notes.
- Lay down a disciplined experiment plan (per `experiment_methodology.md`) that grows the observation space **one feature at a time** without drifting into solver-specific tweaks.

## Finalized Baseline Observation Vector (3D)
This is the official baseline feed to the policy until an experiment explicitly promotes additional features. It keeps only the irreducible signals (time awareness, absolute best fitness, and phase context) so every other metric can be introduced experimentally without combinatorial permutations.

| # | Feature | Signal (Why) | Computation Notes |
|---|---------|--------------|-------------------|
| 1 | `budget_remaining` | Time horizon / urgency. Anchors budget-aware switching. | `(max_evals - current_evals) / max_evals`. |
| 2 | `normalized_best_fitness` | Absolute solution quality, comparable across problems. | `(best - lower) / (upper - lower)` with clipping. |
| 3 | `active_phase` | Lets the policy interpret other signals conditionally (`exploration` vs `exploitation`). | Encode exploration=0.0, exploitation=1.0. |

### Baseline Feature Definitions & Reference Implementations

```python
def compute_budget_remaining(current_evals: int, max_evals: int) -> float:
    """Fraction of evaluation budget left (1.0 → 0.0)."""
    if max_evals <= 0:
        return 1.0
    return 1.0 - (current_evals / float(max_evals))
```

```python
import numpy as np

def compute_normalized_best_fitness(best_fitness: float,
                                    lower_bound: float,
                                    upper_bound: float) -> float:
    """Normalize best fitness to [0, 1] (assumes minimization)."""
    if upper_bound <= lower_bound:
        return 0.0
    norm = (best_fitness - lower_bound) / (upper_bound - lower_bound)
    return float(np.clip(norm, 0.0, 1.0))
```

```python
def compute_active_phase(phase_name: str) -> float:
    """Encode exploration=0.0, exploitation=1.0."""
    return 1.0 if phase_name == "exploitation" else 0.0
```

**Why this particular mix**
- Captures the minimum signals required by every orchestration scenario (budget, absolute quality, phase context).
- Leaves momentum/diversity/structure features outside the baseline so they can be promoted only after proving value.
- Avoids solver-facing metrics or probe costs so it is safe for every registered problem immediately.

## Candidate Feature Backlog (Observation-Only)
The backlog is grouped by incremental cost/complexity. Promote a feature only after an experiment proves statistically significant gains (Wilcoxon vs baseline) *and* runtime overhead <5%.

### Tier 1 — Passive, Low-Cost Augmentations
| Feature | Category | Rationale | Notes |
|---------|----------|-----------|-------|
| `progress_rate` | Dynamics | Momentum check: is the current solver still buying improvements? | `rate_t = α·(prev_norm - curr_norm) + (1-α)·rate_{t-1}`. |
| `population_dispersion` | Diversity | Decision-space spread; flags collapse after exploration. | Mean distance to centroid of normalized representations. |
| `fdc_funnel_proxy` | Structure | Confirm whether distance correlates with fitness (funnel readiness). | Spearman correlation between distance-to-best and fitness. |
| `fitness_entropy` | Fitness distribution | Distinguishes spatial vs objective diversity. | Shannon entropy of normalized fitness histogram. |
| `stagnation_fraction` | Dynamics | Fraction of budget since last improvement; simple stuckness meter. | `(current_evals - last_improvement) / max_evals`. |
| `recent_improvement` | Dynamics | Normalized gain in the *last* step to highlight rare breakthroughs. | `(prev_best - curr_best) / (upper - lower)` clipped ≥0. |
| `uniqueness_ratio` | Diversity | Detects premature convergence even if dispersion stays high. | Ratio of unique encodings in the population. |
| `diversity_collapse_rate` | Dynamics of diversity | EWMA of negative diversity derivative. | Warns of impending collapse earlier than dispersion alone. |

#### Tier‑1 Feature Definitions

```python
def update_progress_rate(current_norm: float,
                         prev_norm: float,
                         prev_rate: float,
                         alpha: float = 0.3) -> float:
    """EWMA improvement velocity clipped to [-1, 1]."""
    delta = prev_norm - current_norm
    new_rate = alpha * delta + (1.0 - alpha) * prev_rate
    return float(np.clip(new_rate, -1.0, 1.0))
```

```python
import numpy as np

def compute_population_dispersion(pop_reps: np.ndarray) -> float:
    """Mean distance to centroid normalized by sqrt(dim)/2."""
    if pop_reps is None or pop_reps.shape[0] < 2:
        return 0.0
    centroid = np.mean(pop_reps, axis=0)
    distances = np.linalg.norm(pop_reps - centroid, axis=1)
    mean_dist = np.mean(distances)
    num_dims = pop_reps.shape[1]
    if num_dims == 0:
        return 0.0
    max_dist = np.sqrt(num_dims) / 2.0
    if max_dist < 1e-9:
        return 0.0
    return float(np.clip(mean_dist / max_dist, 0.0, 1.0))
```

```python
from scipy.stats import spearmanr

def compute_fdc_funnel_proxy(pop_reps: np.ndarray,
                             pop_fits: np.ndarray) -> float:
    """Spearman correlation between distance-to-best and fitness."""
    if pop_reps is None or pop_reps.shape[0] < 3:
        return 0.0
    best_idx = np.argmin(pop_fits)
    best_rep = pop_reps[best_idx]
    distances = np.linalg.norm(pop_reps - best_rep, axis=1)
    try:
        corr, _ = spearmanr(distances, pop_fits)
        if np.isnan(corr):
            return 0.0
        return float(-corr)
    except ValueError:
        return 0.0
```

```python
import numpy as np
from scipy.stats import entropy

def compute_fitness_entropy(pop_fits: np.ndarray,
                            num_bins: int = 10) -> float:
    """Entropy of normalized fitness histogram."""
    if pop_fits is None or len(pop_fits) < 2:
        return 0.0
    min_fit, max_fit = np.min(pop_fits), np.max(pop_fits)
    if max_fit <= min_fit:
        return 0.0
    norm = (pop_fits - min_fit) / (max_fit - min_fit)
    hist, _ = np.histogram(norm, bins=num_bins, range=(0, 1))
    probs = hist / len(pop_fits)
    max_entropy = np.log(num_bins)
    if max_entropy <= 0:
        return 0.0
    return float(entropy(probs, base=np.e) / max_entropy)
```

```python
def compute_stagnation_fraction(current_evals: int,
                                last_improvement_evals: int,
                                max_evals: int) -> float:
    """Budget share spent without any best-fitness improvement."""
    if max_evals <= 0:
        return 0.0
    stagnation = max(0, current_evals - last_improvement_evals)
    return float(stagnation) / float(max_evals)
```

```python
def compute_recent_improvement(prev_best: float,
                               curr_best: float,
                               lower_bound: float,
                               upper_bound: float) -> float:
    """Immediate normalized improvement (≥0)."""
    if upper_bound <= lower_bound:
        return 0.0
    gain = (prev_best - curr_best) / (upper_bound - lower_bound)
    return float(max(0.0, gain))
```

```python
def compute_uniqueness_ratio(pop_reps) -> float:
    """Fraction of unique encodings inside the population."""
    if pop_reps is None or len(pop_reps) < 2:
        return 0.0
    unique = {tuple(rep) for rep in pop_reps}
    return float(len(unique)) / float(len(pop_reps))
```

```python
def compute_diversity_collapse_rate(div_hist: list[float],
                                    alpha: float = 0.5) -> float:
    """EWMA of negative diversity derivative."""
    if len(div_hist) < 2:
        return 0.0
    delta = div_hist[-2] - div_hist[-1]
    rate = alpha * delta + (1.0 - alpha) * 0.0
    return float(np.clip(rate, -1.0, 1.0))
```

### Tier 2 — Structured Population Diagnostics
| Feature | Category | Rationale | Notes |
|---------|----------|-----------|-------|
| `mean_distance_to_best` | Proximity structure | Shows whether the cohort is orbiting the incumbent or exploring multiple basins. | Mean norm of (`rep - best_rep`). |
| `top_k_dispersion` | Elite topology | Detects if elites are scattered (multiple basins) or tight (single basin). | Dispersion computed on top K% individuals. |
| `elite_dominance` | Fitness concentration | Portion of population within ε of best; sharp spike ⇒ ready for switch/terminate. | Already sketched in holographic memo. |
| `fitness_std` | Fitness spread | Cheap variance measure to complement entropy. | Standard deviation of fitness list. |
| `anisotropy` | Shape of occupied space | Captures whether the cloud is spherical or line-like. | Ratio of smallest/largest eigenvalue of covariance. |

#### Tier‑2 Feature Definitions

```python
import numpy as np

def compute_mean_distance_to_best(pop_reps: np.ndarray,
                                  best_rep: np.ndarray) -> float:
    """Mean Euclidean distance from each individual to the best."""
    if pop_reps is None or best_rep is None or pop_reps.shape[0] == 0:
        return 0.0
    distances = np.linalg.norm(pop_reps - best_rep, axis=1)
    return float(np.mean(distances))
```

```python
def compute_top_k_dispersion(pop_reps: np.ndarray,
                             pop_fits: np.ndarray,
                             k_frac: float = 0.1) -> float:
    """Dispersion (mean distance to centroid) for the top K% individuals."""
    if pop_reps is None or pop_reps.shape[0] < 2:
        return 0.0
    N = pop_reps.shape[0]
    k = max(2, int(N * k_frac))
    top_idx = np.argsort(pop_fits)[:k]  # minimization
    elites = pop_reps[top_idx]
    centroid = np.mean(elites, axis=0)
    return float(np.mean(np.linalg.norm(elites - centroid, axis=1)))
```

```python
def compute_elite_dominance(pop_fits: np.ndarray,
                            best_fit: float,
                            lower_bound: float,
                            upper_bound: float,
                            epsilon_frac: float = 0.05) -> float:
    """Fraction within ε of the best normalized fitness."""
    if pop_fits is None or len(pop_fits) == 0:
        return 0.0
    denom = upper_bound - lower_bound if upper_bound > lower_bound else 1.0
    threshold = epsilon_frac * denom
    close = np.sum(np.abs(pop_fits - best_fit) < threshold)
    return float(close) / float(len(pop_fits))
```

```python
def compute_fitness_std(pop_fits) -> float:
    """Standard deviation of population fitness values."""
    if pop_fits is None or len(pop_fits) < 2:
        return 0.0
    return float(np.std(np.asarray(pop_fits, dtype=float)))
```

```python
def compute_anisotropy(pop_reps: np.ndarray) -> float:
    """Ratio of min/max eigenvalues of covariance (shape descriptor)."""
    if pop_reps is None or len(pop_reps) <= pop_reps.shape[1]:
        return 0.0
    centered = pop_reps - np.mean(pop_reps, axis=0)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim < 2:
        return 0.0
    evals = np.linalg.eigvalsh(cov)
    min_ev, max_ev = abs(evals[0]), abs(evals[-1])
    if max_ev < 1e-9:
        return 0.0
    return float(min_ev / max_ev)
```

### Tier 3 — Probe-Based ELA (requires adapter hooks)
| Feature | Category | Rationale | Notes |
|---------|----------|-----------|-------|
| `ruggedness_autocorr` | Landscape smoothness | Lag-1 autocorr along a short cached walk. Smooth ⇒ safe to exploit. | Requires `random_walk` hook and cached fitness trace. |
| `neutrality_walk_proxy` | Plateau detection | Fraction of walk steps with |Δf|<ε. High ⇒ neutral plateaus. | Same walk as above. |
| `deceptiveness_escape_proxy` | Basin escape | Single long jump from best; high value ⇒ current basin is deceptive. | Needs `long_jump` hook + extra eval per probe. |
| `information_content` | Local complexity | Symbolic entropy over walk transitions; merges rugged + neutral cues. | Derived from Tier-3 campaign write-up. |

#### Tier‑3 Feature Definitions

```python
def compute_ruggedness_autocorr(walk_fits: list[float]) -> float:
    """Lag-1 autocorrelation for a random-walk fitness trace."""
    if len(walk_fits) < 2:
        return 0.0
    v = np.array(walk_fits, dtype=float)
    v_demean = v - np.mean(v)
    num = np.dot(v_demean[:-1], v_demean[1:])
    den = np.dot(v_demean, v_demean)
    if den < 1e-9:
        return 1.0
    return float(num / den)
```

```python
def compute_neutrality_walk_proxy(walk_fits: list[float],
                                  epsilon: float = 1e-6) -> float:
    """Fraction of neutral steps along a random walk."""
    if len(walk_fits) < 2:
        return 0.0
    diffs = np.diff(np.asarray(walk_fits, dtype=float))
    neutral = np.sum(np.abs(diffs) < epsilon)
    return float(neutral) / float(len(walk_fits) - 1)
```

```python
def compute_deceptiveness_escape_proxy(curr_best: float,
                                       long_jump_fit: float,
                                       lower_bound: float,
                                       upper_bound: float) -> float:
    """Normalized improvement from a single long-jump probe."""
    if upper_bound <= lower_bound:
        return 0.0
    improvement = (curr_best - long_jump_fit) / (upper_bound - lower_bound)
    return float(np.clip(improvement, 0.0, 1.0))
```

```python
from math import log
from collections import Counter

def compute_information_content(walk_fits: list[float],
                                epsilon: float = 1e-6) -> float:
    """Entropy of symbolized +/-/0 transitions along a walk."""
    if len(walk_fits) < 3:
        return 0.0
    v = np.asarray(walk_fits, dtype=float)
    diffs = np.diff(v)
    symbols = np.zeros_like(diffs, dtype=int)
    symbols[diffs > epsilon] = 1
    symbols[diffs < -epsilon] = -1
    pairs = [(symbols[i], symbols[i + 1])
             for i in range(len(symbols) - 1)
             if symbols[i] != 0 or symbols[i + 1] != 0]
    if not pairs:
        return 0.0
    counts = Counter(pairs)
    total = sum(counts.values())
    probs = np.array(list(counts.values()), dtype=float) / float(total)
    ent = -np.sum(probs * np.log(probs + 1e-12))
    max_ent = np.log(4.0)
    return float(ent / max_ent) if max_ent > 0 else 0.0
```

### Tier 4 — Advanced / Holographic Modeling
| Feature | Category | Rationale | Notes |
|---------|----------|-----------|-------|
| `local_ruggedness` | Texture from motion | Ratio of |Δfitness| to |Δcentroid|; interprets solver wake instead of probes. | Needs centroid history buffers. |
| `path_coherence` | Momentum alignment | Cosine between successive centroid steps to detect thrashing vs purposeful motion. | Requires stored centroid trajectory. |
| `linear_r2` | Surrogate fit quality | Measures how well a linear model explains local landscape; high ⇒ deterministic slope. | Requires solving least squares each step. |
| `information_gain_rate` | Meta-signal | Rate of change of log reward contributions per observation dimension; helps judge diminishing returns of adding features. | Computed from logged training statistics, no solver introspection. |

#### Tier‑4 Feature Definitions

```python
def compute_local_ruggedness(centroid_history: list[np.ndarray],
                             best_fit_history: list[float]) -> float:
    """Sigmoid(|Δfitness| / |Δcentroid|) using last two steps."""
    if len(centroid_history) < 2 or len(best_fit_history) < 2:
        return 0.0
    dist = np.linalg.norm(centroid_history[-1] - centroid_history[-2])
    d_fit = abs(best_fit_history[-1] - best_fit_history[-2])
    if dist < 1e-9:
        return 0.0
    ratio = d_fit / dist
    return float(2.0 * (1.0 / (1.0 + np.exp(-ratio))) - 1.0)
```

```python
def compute_path_coherence(centroid_history: list[np.ndarray]) -> float:
    """Cosine similarity between successive centroid movement vectors."""
    if len(centroid_history) < 3:
        return 0.0
    v_t = centroid_history[-1] - centroid_history[-2]
    v_prev = centroid_history[-2] - centroid_history[-3]
    norm_t = np.linalg.norm(v_t)
    norm_prev = np.linalg.norm(v_prev)
    if norm_t < 1e-9 or norm_prev < 1e-9:
        return 0.0
    return float(np.dot(v_t, v_prev) / (norm_t * norm_prev))
```

```python
def compute_linear_r2(pop_reps: np.ndarray,
                      pop_fits: np.ndarray) -> float:
    """R^2 of a linear regression predicting fitness from coordinates."""
    if pop_reps is None or pop_reps.shape[0] < 2:
        return 0.0
    X = pop_reps.astype(float)
    y = np.asarray(pop_fits, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    Xb = np.hstack([ones, X])
    try:
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    y_pred = Xb @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
```

```python
def compute_information_gain_rate(log_rewards: list[float],
                                  obs_dim: int) -> float:
    """
    Heuristic meta-signal: slope of recent log reward averages divided
    by observation dimensionality. Useful for spotting diminishing returns.
    """
    if len(log_rewards) < 2 or obs_dim <= 0:
        return 0.0
    window = np.asarray(log_rewards[-20:], dtype=float)
    slope = (window[-1] - window[0]) / max(1, len(window) - 1)
    return float(slope / obs_dim)
```

## Observation-Space Experiment Roadmap
Every iteration follows the same recipe: start from the current baseline, add exactly one feature, run the full train/eval loop (train on NKL/MaxCut/Knapsack, evaluate on NKL/MaxCut/Knapsack/TSP), and decide whether to promote the feature. Categories only indicate compute cost—the iteration itself touches every candidate whose prerequisites are satisfied.

### Step 0 — Baseline Control Run (Iteration 1)
1. Train PPO on NKL/MaxCut/Knapsack with the 3D baseline (no additions) for a fixed horizon of **6,000,000 agent decisions** (identical across all runs). Each training episode randomizes problem parameters (instance sizes/difficulties), selects a random but valid solver pair, and samples a logically consistent search budget; the chosen configuration remains fixed within that trajectory to keep the RL rollout consistent. Training configuration (parallel env count, PPO batch size, learning rate, clip range, reward logging flags, etc.) must be recorded in `config.yaml` and kept identical across the baseline + all candidates within the iteration.
2. Evaluate the resulting policy on NKL, MaxCut, Knapsack, and TSP using a fixed schedule of **1,000 evaluation episodes per problem** (with predetermined instances/seeds). Archive all metrics (reward traces, best-fitness distributions, switch histograms) along with the per-episode sampled budgets/solver pairs used in evaluation.
3. This run becomes the control dataset for the first iteration.

### Iterative Feature Cycle
For each iteration `i` (starting at 1):
1. **Candidate queue**: traverse the entire backlog (Tier 1 → Tier 2 → Tier 3 → Tier 4), skipping only features whose prerequisites (adapter hooks, historical buffers) are not yet available.
2. **Single feature injection**: extend the current baseline observation vector with exactly one candidate (baseline dimension + 1).
3. **Training**: retrain PPO on NKL/MaxCut/Knapsack using the same seeds/hyperparameters as the control run for that iteration. Per-episode randomization (problem parameters, solver pairs, budget) mirrors the baseline: sample once at episode start and hold constant for that trajectory. Log the sampled parameters/budgets/solver IDs per episode so later analysis can reconstruct the training distribution.
4. **Evaluation**: run the trained policy on NKL, MaxCut, Knapsack, and TSP for 1,000 episodes per problem (same instance/seed roster as the control). Capture the same metrics as the control and log the sampled solver pairs/budgets used during evaluation.
5. **Decision**: compare against the control via paired statistical tests and the promotion criteria listed below. Promote the feature only if it meets every criterion; otherwise record the failure and fall back to the current baseline.
6. **Regression check**: after an iteration finishes and a best configuration is chosen, run a quick regression (baseline vs. “baseline + promoted features”) under the same seeds to confirm the cumulative stack still satisfies every criterion before declaring the new baseline final.
5. **Decision**: compare against the control via paired statistical tests and the promotion criteria listed below. Promote the feature only if it meets every criterion; otherwise record the failure and fall back to the current baseline.
6. **Baseline update**: once every candidate in the backlog has been tested for iteration `i`, select the best-performing configuration observed in that iteration (which may still be the previous baseline) as the starting point for iteration `i+1`. If new prerequisites become available (e.g., probe hooks, historical buffers), add the corresponding features to the queue for the next iteration.

### Promotion Criteria (Effectiveness + Efficiency)
Every candidate must satisfy **all** of the following:

| Criterion | Description |
|-----------|-------------|
| **Normalized per-problem gains** | Compute normalized best-fitness (or % improvement vs. fixed references) for each NKL, MaxCut, Knapsack, and TSP instance. Use paired Wilcoxon signed-rank tests (two-sided, α = 0.05 with Holm–Bonferroni correction across the four problems) over identical seeds. Candidate must improve or tie the baseline on trained problems and stay within ≤1 % regression on TSP. |
| **Behavioral quality** | Compare phase-switch distributions (mean/variance/histograms) and inspect qualitative traces. Reject features that elongate runs or induce oscillations even if fitness improves. |
| **Held-out generalization** | TSP evaluation (never seen in training) must at least match the baseline’s normalized median reward/fitness. Also test unseen NKL/MaxCut/Knapsack instances to ensure gains aren’t instance-specific. |
| **Observation-time budget** | Instrument `ObservationComputer` (and adapter hooks) to measure ms/step. Additional observation cost must remain ≤5 % of the baseline cost. Probe-driven features must charge their evaluation time against this budget. |
| **Policy (DRL) overhead** | Measure PPO forward/backward time separately from solver time. Larger observation tensors must not slow policy updates or inference by more than ~5 %. |
| **Solver-independent timing** | Log solver runtime separately so observation/policy overhead is isolated. Promotions reference only the controllable overhead buckets. |
| **Wall-clock & resource limits** | Track total training hours, evaluation FPS, and GPU/CPU/memory usage. Features exceeding the agreed runtime/resource budget (~5 %) fail even if accuracy improves. |
| **Seed robustness** | Use ≥3 seeds per configuration. No single seed may regress by >5 % normalized fitness relative to the control. |
| **Reward-component logging** | `RewardComputer` must emit per-component contributions per episode so behavioral changes can be attributed correctly. Missing reward breakdowns invalidate the run. |
| **Complete metrics bundle** | Promotion requires the full log set (normalized curves, rewards, switch histograms, timing tables). Missing data disqualifies the feature. |
| **Probe evaluation accounting** | If a feature uses probes/random walks/long jumps, their evaluation cost must be deducted from the episode’s budget counter and logged explicitly (counts + time). Probe sampling must never mutate the live population state. |
| **Implementation integrity** | Feature must integrate without solver-specific hacks; probes must not mutate live populations or violate budget accounting. Prerequisites (hooks, history buffers) must be in place before testing. |
| **Regression verification** | After an iteration’s promotions, rerun the baseline vs. “baseline + promoted features” comparison to ensure the combined stack still satisfies all criteria. |
| **Documentation & traceability** | Log feature name, config hash, timing stats, statistical results, and decision outcome after every run. Update the baseline table and `RLOrchestrator/core/observation.py` comments whenever the canonical schema changes. |
| **Retirement policy** | Features that fail multiple iterations (e.g., two attempts) are retired until upstream changes justify revisiting them, keeping queues lean. |

### Documentation & Traceability
- Log every training/evaluation pair with its feature name, seeds, metrics, solver-pair statistics, timing breakdowns, and promotion decision in the experiment log.
- When a feature becomes part of the official baseline, update the top-of-file baseline table plus the code comments in `RLOrchestrator/core/observation.py` so the canonical schema stays in sync.
- Retire candidates that repeatedly fail promotion so later iterations focus on the most promising signals.
- Persist artifacts using the structure:
  ```
  logs/observation_experiments/
      iteration_<N>/
          baseline/
              config.yaml
              solver_pairs_train.csv
              solver_pairs_eval.csv
              training_metrics.csv
              eval_metrics.csv
              timing_breakdown.json
              plots/
          feature_<name>/
              config.yaml
              solver_pairs_train.csv
              solver_pairs_eval.csv
              training_metrics.csv
              eval_metrics.csv
              timing_breakdown.json
              plots/
  ```
  Each `config.yaml` records seeds, hyperparameters, observation schema, sampled budget ranges, normalization references (per-instance lower/upper bounds or baseline heuristics), and the git commit SHA used for the run. `solver_pairs_train.csv`/`solver_pairs_eval.csv` capture the solver combinations and counts observed during training/evaluation episodes. `training_metrics.csv`/`eval_metrics.csv` contain per-problem normalized stats and raw values; `timing_breakdown.json` stores observation/policy/solver times and resource usage; `plots/` holds generated charts (switch histograms, reward curves). A run without a complete folder (or with inconsistent naming) is invalid and must be repeated. All experiment directories must be retained (and backed up) for the life of the project to support future audits and analyses.

This playbook keeps all observation-space decisions centralized, auditable, and detached from solver minutiae while providing a concrete path for expanding the observation space responsibly.
