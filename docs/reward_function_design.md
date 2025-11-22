# PMSP v3: Phase-Modulated Signal Processing Reward

Mathematically rigorous reward for "Effectiveness First, Efficiency Later" with bounded, smooth signals tailored to PPO/SGD stability.

---

## Mathematical Requirements for Deep RL

- **Critic learnability:** Bounded returns, low variance, step/terminal on same scale.
- **Policy gradient signal:** Non-sparse, informative variance, Lipschitz smoothness.
- **Numerical stability:** |r| < 100, no divide-by-zero, smooth activations.

---

## Specification (Brief)

- Effectiveness: log-improvement, phase-gated diversity in exploration.
- Efficiency: sigmoid urgency × stagnation with smooth grace after switches.
- Smoothing: EMA to reduce variance and encode trend.
- Terminal: bounded final quality bonus; defaults keep return variance low.

---

## Mathematical Validation (Key Points)

- **Boundedness:** default `r_step ∈ (-1.25, 1)`, `r_total ∈ (-1.25, 3.5)`.
- **Lipschitz:** main path is log → scale → tanh; Lipschitz constant ~ `gain_sensitivity/ε` (practically ~k for fitness > 1e-3).
- **Variance:** terminal weight `w_term ≤ 2.5` keeps Var(G) moderate; pressure bounded by `pressure_max`.
- **Gradient flow:** all components use tanh/sigmoid/cosine; no step functions.

---

## Scenario Validation Matrix (Summary)

- Startup/Trust Fund: urgency + EMA stagnation; smooth, bounded pressure.
- Sudden Death: dense step rewards + terminal anchor.
- Needle/Haystack: diversity term keeps gradient when gain=0.
- Honey Pot: terminal dominates long-term optimum.
- Rugged/Funnel: EMA smooths noise; log-space rewards small late gains.
- Cold Start: grace window smooth (cosine/linear).
- Busy Fool: diversity gated to exploration + terminal truth.
- Dip/Late Bloomer/One-Hit: ignore regressions, long EMA memory, pressure after signal fades.

---

## Code

```python
"""
PMSP v3: Phase-Modulated Signal Processing Reward
Mathematically Rigorous Design for Deep Reinforcement Learning

Design Constraints Satisfied:
1. All rewards bounded to [-1, +1] base range
2. Lipschitz continuous in all inputs
3. Return variance controlled via terminal/step ratio
4. No discontinuities or step functions
5. Numerically stable for float32

Author: Research Team
Version: 3.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class PMSPConfigV3:
    """
    Immutable configuration with mathematical constraints documented.
    
    Theoretical Bounds:
        r_step ∈ [-1, +1] × global_scale
        r_terminal ∈ [0, w_term] × global_scale
        r_total ∈ [-1, 1 + w_term] × global_scale
    
    For episode length T with γ=0.99:
        G_t ∈ [-T, T + w_term] × global_scale (approximately)
        Var(G_t) scales with w_term² (keep w_term small)
    """
    
    # === GLOBAL SCALING ===
    global_scale: float = 1.0
    
    # === EFFECTIVENESS (Primary Signal) ===
    # Controls reward for fitness improvement
    # Mathematical role: Provides dense gradient signal
    gain_sensitivity: float = 3.0       # Scaling inside tanh (controls gradient steepness)
    
    # === DIVERSITY (Exploration Bonus) ===
    # Only active in exploration phase
    # Mathematical role: Prevents policy collapse to single action
    div_weight: float = 0.15            # Relative weight vs improvement
    div_phase_gate: bool = True         # Only reward diversity in phase 0
    
    # === EFFICIENCY PRESSURE ===
    # Penalizes stagnation, scaled by budget urgency
    # Mathematical role: Shapes value function to prefer termination when stuck
    pressure_max: float = 0.25          # Maximum pressure (asymmetric: reward > penalty)
    pressure_onset: float = 0.5         # Budget ratio where pressure activates
    pressure_steepness: float = 5.0     # Sigmoid steepness at onset
    
    # === TEMPORAL SMOOTHING ===
    # EMA parameters for noise reduction
    # Mathematical role: Reduces variance in advantage estimates
    ema_alpha: float = 0.12             # Higher = more reactive, Lower = more stable
    ema_initial: float = 0.2            # Optimistic prior (assume competence)
    
    # === PHASE TRANSITION HANDLING ===
    # Grace period prevents penalizing solver warm-up
    # Mathematical role: Removes discontinuity at phase boundaries
    grace_steps: int = 10               # Steps of immunity after switch
    grace_curve: str = "cosine"         # "linear" or "cosine" interpolation
    
    # === TERMINAL REWARD ===
    # Final quality assessment
    # Mathematical role: Provides ground truth signal, anchors value function
    # CRITICAL: Keep ≤ 5× step reward to limit return variance
    w_term: float = 2.5                 # Terminal multiplier
    term_shaping: str = "linear"        # "linear", "sqrt", or "log"
    
    # === NUMERICAL STABILITY ===
    eps: float = 1e-8                   # Floor for divisions and logs
    clip_signal: float = 4.0            # Pre-tanh clipping (gradient control)


class PMSPRewardV3:
    """
    PMSP v3: Gradient-Friendly Phase-Modulated Reward Computer.
    
    Mathematical Properties:
    -------------------------
    1. BOUNDEDNESS: All outputs in [-1-ε, 1+w_term] × global_scale
    
    2. LIPSCHITZ CONTINUITY: 
       |r(s₁) - r(s₂)| ≤ L × |s₁ - s₂|
       where L ≈ gain_sensitivity × global_scale
       
    3. SMOOTHNESS:
       All components use tanh, sigmoid, or cosine
       No step functions or discontinuities
       
    4. VARIANCE CONTROL:
       Var(G_t) ≈ O(T × global_scale²) + O(w_term² × global_scale²)
       Keeping w_term ≤ 3 ensures terminal doesn't dominate
    
    Scenario Coverage:
    ------------------
    See validation matrix in documentation.
    """

    def __init__(self, config: Optional[PMSPConfigV3] = None):
        self.cfg = config or PMSPConfigV3()
        self._validate_config()
        self._init_state()

    def _validate_config(self):
        """Ensure configuration satisfies mathematical constraints."""
        c = self.cfg
        assert 0 < c.ema_alpha < 1, "EMA alpha must be in (0, 1)"
        assert c.pressure_max >= 0, "Pressure must be non-negative"
        assert c.w_term >= 0, "Terminal weight must be non-negative"
        assert c.gain_sensitivity > 0, "Gain sensitivity must be positive"
        assert c.grace_steps >= 0, "Grace steps must be non-negative"
        assert c.pressure_onset > 0, "Pressure onset must be positive"
        
        # Warn if likely to cause training instability
        if c.w_term > 5.0:
            import warnings
            warnings.warn(
                f"w_term={c.w_term} > 5.0 may cause high return variance. "
                "Consider reducing for stable Critic learning."
            )

    def _init_state(self):
        """Initialize episode state variables."""
        self._prev_fitness: float = 1.0
        self._ema_signal: float = self.cfg.ema_initial
        self._prev_phase: int = -1
        self._steps_since_switch: int = 0
        self._step_count: int = 0

    def reset(self, initial_fitness: float = 1.0):
        """
        Reset state for new episode.
        
        Args:
            initial_fitness: Starting normalized fitness (lower = better)
        """
        self._prev_fitness = np.clip(initial_fitness, self.cfg.eps, 1.0)
        self._ema_signal = self.cfg.ema_initial
        self._prev_phase = -1
        self._steps_since_switch = 0
        self._step_count = 0

    def compute(
        self,
        observation: np.ndarray,
        evals_used: int,
        total_budget: int,
        terminated: bool,
    ) -> float:
        """
        Compute reward for current transition.
        
        Mathematical Guarantee:
            output ∈ [-1.25, 3.5] × global_scale (with default config)
        
        Args:
            observation: State vector
                [0]: budget_ratio ∈ [0, 1], 1 = full budget remaining
                [1]: normalized_fitness ∈ [0, 1], lower = better
                [4]: diversity ∈ [0, 1], higher = more diverse
                [5]: phase ∈ {0, 1}, 0 = exploration, 1 = exploitation
            evals_used: Evaluations consumed this step
            total_budget: Maximum evaluation budget
            terminated: Episode termination flag
            
        Returns:
            Scalar reward value
        """
        cfg = self.cfg
        self._step_count += 1

        # ============================================================
        # STAGE 1: PARSE OBSERVATION (with numerical safety)
        # ============================================================
        budget_ratio = float(np.clip(observation[0], 0.0, 1.0))
        curr_fitness = float(np.clip(observation[1], cfg.eps, 1.0))
        curr_diversity = float(np.clip(observation[4], 0.0, 1.0))
        curr_phase = int(observation[5])

        # ============================================================
        # STAGE 2: PHASE TRANSITION DETECTION
        # ============================================================
        if curr_phase != self._prev_phase and self._prev_phase != -1:
            self._steps_since_switch = 0
        else:
            self._steps_since_switch += 1

        # ============================================================
        # STAGE 3: COMPUTE IMPROVEMENT SIGNAL
        # ============================================================
        log_prev = np.log(self._prev_fitness + cfg.eps)
        log_curr = np.log(curr_fitness + cfg.eps)
        log_improvement = log_prev - log_curr  # Positive when fitness improves
        gain = np.maximum(0.0, log_improvement)

        # ============================================================
        # STAGE 4: PHASE-MODULATED SIGNAL COMPOSITION
        # ============================================================
        if curr_phase == 0 and cfg.div_phase_gate:
            raw_signal = gain + cfg.div_weight * curr_diversity
        else:
            raw_signal = gain

        scaled_signal = np.clip(
            cfg.gain_sensitivity * raw_signal,
            -cfg.clip_signal,
            cfg.clip_signal,
        )

        # ============================================================
        # STAGE 5: TEMPORAL SMOOTHING (EMA)
        # ============================================================
        self._ema_signal = (
            (1.0 - cfg.ema_alpha) * self._ema_signal +
            cfg.ema_alpha * scaled_signal
        )

        # ============================================================
        # STAGE 6: COMPUTE EFFICIENCY PRESSURE
        # ============================================================
        urgency_input = (cfg.pressure_onset - budget_ratio) * cfg.pressure_steepness
        urgency = self._stable_sigmoid(urgency_input)
        stagnation = 1.0 - np.tanh(np.maximum(0.0, self._ema_signal))

        if self._steps_since_switch < cfg.grace_steps:
            t = self._steps_since_switch / max(1, cfg.grace_steps)
            if cfg.grace_curve == "cosine":
                grace_factor = 0.5 * (1.0 - np.cos(np.pi * t))
            else:
                grace_factor = t
        else:
            grace_factor = 1.0

        pressure = cfg.pressure_max * urgency * stagnation * grace_factor

        # ============================================================
        # STAGE 7: COMPUTE EFFECTIVENESS REWARD
        # ============================================================
        r_effectiveness = np.tanh(scaled_signal)

        # ============================================================
        # STAGE 8: COMPUTE TERMINAL REWARD
        # ============================================================
        r_terminal = 0.0
        if terminated:
            final_quality = 1.0 - curr_fitness
            if cfg.term_shaping == "sqrt":
                shaped_quality = np.sqrt(final_quality)
            elif cfg.term_shaping == "log":
                shaped_quality = np.log1p(final_quality) / np.log(2)
            else:
                shaped_quality = final_quality
            r_terminal = cfg.w_term * shaped_quality

        # ============================================================
        # STAGE 9: ASSEMBLE FINAL REWARD
        # ============================================================
        r_step = r_effectiveness - pressure
        r_total = (r_step + r_terminal) * cfg.global_scale

        # ============================================================
        # STAGE 10: UPDATE STATE
        # ============================================================
        self._prev_fitness = curr_fitness
        self._prev_phase = curr_phase

        return float(r_total)

    @staticmethod
    def _stable_sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)

    def get_diagnostics(self) -> Dict[str, float]:
        """Return internal state for logging and debugging."""
        return {
            "ema_signal": float(self._ema_signal),
            "prev_fitness": float(self._prev_fitness),
            "steps_since_switch": int(self._steps_since_switch),
            "step_count": int(self._step_count),
        }

    @staticmethod
    def get_theoretical_bounds(config: PMSPConfigV3) -> Dict[str, float]:
        """Compute theoretical reward bounds for a given configuration."""
        r_step_min = -1.0 - config.pressure_max
        r_step_max = 1.0
        r_total_min = r_step_min * config.global_scale
        r_total_max = (r_step_max + config.w_term) * config.global_scale
        return {
            "r_step_min": r_step_min,
            "r_step_max": r_step_max,
            "r_total_min": r_total_min,
            "r_total_max": r_total_max,
            "theoretical_range": r_total_max - r_total_min,
        }


# ================================================================
# PRESET CONFIGURATIONS
# ================================================================

def config_stable() -> PMSPConfigV3:
    """Conservative configuration for initial experiments."""
    return PMSPConfigV3(
        global_scale=1.0,
        gain_sensitivity=2.5,
        div_weight=0.12,
        pressure_max=0.15,
        pressure_onset=0.4,
        pressure_steepness=4.0,
        ema_alpha=0.08,
        grace_steps=12,
        grace_curve="cosine",
        w_term=2.0,
        term_shaping="linear",
    )


def config_balanced() -> PMSPConfigV3:
    """Balanced configuration for typical problems."""
    return PMSPConfigV3(
        global_scale=1.0,
        gain_sensitivity=3.0,
        div_weight=0.15,
        pressure_max=0.25,
        pressure_onset=0.5,
        pressure_steepness=5.0,
        ema_alpha=0.12,
        grace_steps=10,
        grace_curve="cosine",
        w_term=2.5,
        term_shaping="linear",
    )


def config_sparse() -> PMSPConfigV3:
    """Configuration for sparse/deceptive fitness landscapes."""
    return PMSPConfigV3(
        global_scale=1.0,
        gain_sensitivity=2.0,
        div_weight=0.35,
        pressure_max=0.10,
        pressure_onset=0.3,
        pressure_steepness=3.0,
        ema_alpha=0.05,
        grace_steps=15,
        grace_curve="cosine",
        w_term=2.0,
        term_shaping="sqrt",
    )


def config_fast_converge() -> PMSPConfigV3:
    """Configuration for well-behaved unimodal problems."""
    return PMSPConfigV3(
        global_scale=1.0,
        gain_sensitivity=4.0,
        div_weight=0.08,
        pressure_max=0.35,
        pressure_onset=0.6,
        pressure_steepness=6.0,
        ema_alpha=0.18,
        grace_steps=6,
        grace_curve="linear",
        w_term=3.0,
        term_shaping="linear",
    )
```
