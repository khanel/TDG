#!/usr/bin/env python3
"""
Episode Simulation Validator for ISR v2.0
==========================================

Tests complete episode trajectories to verify cumulative rewards make sense.
This validates the reward function at the STRATEGIC level - does it incentivize
the behaviors we want over full episodes?

Design Philosophy:
-----------------
Rather than testing individual reward scenarios, we simulate COMPLETE EPISODES
and verify that the total cumulative reward matches our expectations:

1. Good Strategies should have HIGHER cumulative rewards than bad strategies
2. Early exit should be penalized (unless quality is excellent)
3. Pure exploration without switch should be penalized (budget waste)
4. Pure exploitation without exploration should be penalized (no diversity)
5. Balanced strategies should have highest rewards

Episode Scenarios:
-----------------
- EARLY_EXIT_LOW_QUALITY: Exit immediately with poor solution
- EARLY_EXIT_HIGH_QUALITY: Exit immediately with excellent solution (lucky!)
- ALL_EXPLORATION: Never switch to exploitation, waste budget exploring
- ALL_EXPLOITATION: Skip exploration, go straight to exploitation
- BALANCED_GOOD: Proper exploration → exploitation → termination with improvements
- BALANCED_STAGNANT: Proper timing but no improvements found
- LATE_SWITCH: Stay in exploration too long, little time for exploitation
- PREMATURE_SWITCH: Switch while still finding improvements

Author: Validation Framework
Version: 1.0.0
"""

import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from temp.core.intelligent_reward import IntelligentSearchReward, ISRConfig, RewardSignal


class Phase(Enum):
    EXPLORATION = "EXPLORATION"
    EXPLOITATION = "EXPLOITATION"
    TERMINATION = "TERMINATION"


@dataclass
class StepRecord:
    """Record of a single episode step."""
    step: int
    phase: Phase
    action: int
    improvement: int
    diversity: float
    stagnation: float
    quality: float
    budget: float
    reward: float
    components: Dict[str, float]


@dataclass
class EpisodeResult:
    """Result of a simulated episode."""
    name: str
    description: str
    steps: List[StepRecord]
    total_reward: float
    final_quality: float
    final_phase: Phase
    explore_steps: int
    exploit_steps: int
    expected_ranking: str  # "high", "medium", "low"
    
    def summary(self) -> str:
        return (
            f"[{self.expected_ranking.upper():6s}] {self.name}: "
            f"Total={self.total_reward:+.3f}, Quality={self.final_quality:.2f}, "
            f"Explore={self.explore_steps}, Exploit={self.exploit_steps}"
        )


class EpisodeSimulator:
    """
    Simulates complete episodes with the ISR reward function.
    """
    
    def __init__(self, config: Optional[ISRConfig] = None, max_steps: int = 100):
        self.reward_fn = IntelligentSearchReward(config)
        self.max_steps = max_steps
    
    def simulate_step(
        self,
        phase: Phase,
        action: int,
        improvement: int,
        diversity: float,
        stagnation: float,
        quality: float,
        budget: float
    ) -> RewardSignal:
        """Simulate a single step and get reward."""
        state = {
            'improvement_binary': improvement,
            'diversity_score': diversity,
            'stagnation_ratio': stagnation,
            'fitness_norm': 1.0 - quality,  # fitness_norm is inverse of quality
            'budget_consumed': budget,
        }
        context = {'current_phase': phase.value}
        return self.reward_fn.calculate(state, action, context)
    
    def run_episode(
        self,
        name: str,
        description: str,
        trajectory: List[Tuple[Phase, int, int, float, float, float]],
        expected_ranking: str
    ) -> EpisodeResult:
        """
        Run a complete episode simulation.
        
        Args:
            name: Episode name
            description: What this episode tests
            trajectory: List of (phase, action, improvement, diversity, stagnation, quality)
                        Budget is computed automatically from step count
            expected_ranking: "high", "medium", "low" - expected relative performance
        
        Returns:
            EpisodeResult with full breakdown
        """
        steps = []
        total_reward = 0.0
        explore_steps = 0
        exploit_steps = 0
        
        for i, (phase, action, improvement, diversity, stagnation, quality) in enumerate(trajectory):
            budget = (i + 1) / self.max_steps
            
            result = self.simulate_step(
                phase, action, improvement, diversity, stagnation, quality, budget
            )
            
            step_record = StepRecord(
                step=i + 1,
                phase=phase,
                action=action,
                improvement=improvement,
                diversity=diversity,
                stagnation=stagnation,
                quality=quality,
                budget=budget,
                reward=result.total_value,
                components=result.raw_components
            )
            steps.append(step_record)
            total_reward += result.total_value
            
            if phase == Phase.EXPLORATION:
                explore_steps += 1
            elif phase == Phase.EXPLOITATION:
                exploit_steps += 1
        
        final_quality = trajectory[-1][5] if trajectory else 0.0
        final_phase = trajectory[-1][0] if trajectory else Phase.EXPLORATION
        
        return EpisodeResult(
            name=name,
            description=description,
            steps=steps,
            total_reward=total_reward,
            final_quality=final_quality,
            final_phase=final_phase,
            explore_steps=explore_steps,
            exploit_steps=exploit_steps,
            expected_ranking=expected_ranking
        )


def create_episode_scenarios(max_steps: int = 100) -> List[Tuple[str, str, List, str]]:
    """
    Create a comprehensive set of episode scenarios to validate.
    
    Each scenario is: (name, description, trajectory, expected_ranking)
    Trajectory: List of (phase, action, improvement, diversity, stagnation, quality)
    
    Rankings: "high" should have better rewards than "medium" > "low"
    """
    scenarios = []
    
    # =========================================================================
    # SCENARIO 1: EARLY EXIT LOW QUALITY
    # Agent advances through phases immediately without doing any real search
    # Should have VERY LOW reward - this is the behavior we're seeing!
    # =========================================================================
    trajectory = [
        # 1 step exploration, advance immediately
        (Phase.EXPLORATION, 1, 0, 0.3, 0.1, 0.2),  # Premature switch (low stagnation)
        # 1 step exploitation, advance immediately
        (Phase.EXPLOITATION, 1, 0, 0.2, 0.2, 0.25),  # Premature advance
        # Termination with poor quality
        (Phase.TERMINATION, 0, 0, 0.2, 0.2, 0.25),
    ]
    scenarios.append((
        "EARLY_EXIT_LOW_QUALITY",
        "Exit immediately with poor quality (THE BUG BEHAVIOR)",
        trajectory,
        "low"
    ))
    
    # =========================================================================
    # SCENARIO 2: EARLY EXIT HIGH QUALITY (lucky scenario)
    # Agent advances quickly but somehow has excellent solution
    # Should still be penalized for premature transitions, but termination reward is high
    # =========================================================================
    trajectory = [
        # 1 step exploration with improvement
        (Phase.EXPLORATION, 1, 1, 0.5, 0.1, 0.9),  # SWITCH while improving! Bad
        # 1 step exploitation 
        (Phase.EXPLOITATION, 1, 0, 0.3, 0.5, 0.92),  # Advance with OK stagnation
        # Termination with excellent quality
        (Phase.TERMINATION, 0, 0, 0.3, 0.5, 0.92),
    ]
    scenarios.append((
        "EARLY_EXIT_HIGH_QUALITY",
        "Exit immediately but with excellent quality (lucky start)",
        trajectory,
        "medium"  # Good quality saves it somewhat
    ))
    
    # =========================================================================
    # SCENARIO 3: ALL EXPLORATION NO SWITCH
    # Agent never switches to exploitation, wastes entire budget exploring
    # Should have MEDIUM-LOW reward (gets diversity bonuses but no refinement)
    # =========================================================================
    explore_steps = 80
    trajectory = []
    for i in range(explore_steps):
        # Alternating improvements and stagnation, high diversity
        improvement = 1 if i % 5 == 0 else 0
        stagnation = 0.2 + (i / explore_steps) * 0.5  # Increasing stagnation
        diversity = 0.7 - (i / explore_steps) * 0.2  # Decreasing diversity
        quality = 0.3 + (i / explore_steps) * 0.2  # Slowly improving
        trajectory.append((Phase.EXPLORATION, 0, improvement, diversity, stagnation, quality))
    
    # Eventually forced to switch due to budget exhaustion (simulated)
    trajectory.append((Phase.EXPLORATION, 1, 0, 0.4, 0.7, 0.5))  # Switch at high stagnation
    trajectory.append((Phase.EXPLOITATION, 1, 0, 0.3, 0.6, 0.52))  # Quick advance
    trajectory.append((Phase.TERMINATION, 0, 0, 0.3, 0.6, 0.52))
    
    scenarios.append((
        "ALL_EXPLORATION_LATE_SWITCH",
        "Explore 80% of budget, barely any exploitation",
        trajectory,
        "low"  # Wasted budget on exploration
    ))
    
    # =========================================================================
    # SCENARIO 4: ALL EXPLOITATION NO EXPLORATION
    # Agent switches immediately, spends all time in exploitation
    # Should have MEDIUM-LOW reward (no diversity benefit, may get stuck)
    # =========================================================================
    trajectory = [
        # Single exploration step then switch
        (Phase.EXPLORATION, 1, 0, 0.3, 0.1, 0.2),  # Premature switch
    ]
    
    # 50 exploitation steps
    for i in range(50):
        improvement = 1 if i < 5 else 0  # Initial improvements then stagnation
        stagnation = min(1.0, 0.1 + i * 0.02)
        quality = min(0.6, 0.2 + i * 0.01)  # Quickly plateaus
        trajectory.append((Phase.EXPLOITATION, 0, improvement, 0.2, stagnation, quality))
    
    # Termination
    trajectory.append((Phase.EXPLOITATION, 1, 0, 0.15, 0.9, 0.6))
    trajectory.append((Phase.TERMINATION, 0, 0, 0.15, 0.9, 0.6))
    
    scenarios.append((
        "ALL_EXPLOITATION_PREMATURE_SWITCH",
        "Skip exploration, all exploitation - gets stuck",
        trajectory,
        "medium"  # Some quality achieved
    ))
    
    # =========================================================================
    # SCENARIO 5: BALANCED GOOD - THE IDEAL STRATEGY
    # Proper exploration (20 steps) → exploitation (30 steps) → termination
    # With good improvements in both phases
    # Should have HIGH reward
    # =========================================================================
    trajectory = []
    
    # Exploration phase: 20 steps, good improvements, high diversity
    for i in range(20):
        improvement = 1 if i % 3 == 0 else 0  # Frequent improvements
        diversity = 0.8 - (i / 20) * 0.2  # Starts high, decreases
        stagnation = min(0.5, i * 0.025)  # Slowly increasing
        quality = 0.2 + (i / 20) * 0.25  # Building quality
        trajectory.append((Phase.EXPLORATION, 0, improvement, diversity, stagnation, quality))
    
    # Switch when stagnation is moderate and no improvement
    trajectory.append((Phase.EXPLORATION, 1, 0, 0.5, 0.5, 0.45))  # Good switch timing
    
    # Exploitation phase: 30 steps, refining
    for i in range(30):
        improvement = 1 if i % 4 == 0 else 0  # Regular refinements
        stagnation = 0.2 + (i / 30) * 0.5
        quality = 0.45 + (i / 30) * 0.4  # Significant improvement
        trajectory.append((Phase.EXPLOITATION, 0, improvement, 0.3, stagnation, quality))
    
    # Advance to termination when stagnant
    trajectory.append((Phase.EXPLOITATION, 1, 0, 0.25, 0.75, 0.85))  # Good timing
    trajectory.append((Phase.TERMINATION, 0, 0, 0.25, 0.75, 0.85))
    
    scenarios.append((
        "BALANCED_GOOD",
        "Ideal: proper explore/exploit balance with improvements",
        trajectory,
        "high"
    ))
    
    # =========================================================================
    # SCENARIO 6: BALANCED BUT STAGNANT
    # Proper timing but no improvements found (hard problem)
    # Should have MEDIUM reward (good process, bad luck)
    # =========================================================================
    trajectory = []
    
    # Exploration: 20 steps, minimal improvements
    for i in range(20):
        improvement = 1 if i == 5 else 0  # Only 1 improvement
        diversity = 0.6 - (i / 20) * 0.3
        stagnation = min(0.7, i * 0.035)
        quality = 0.15 + (i / 20) * 0.1  # Barely improving
        trajectory.append((Phase.EXPLORATION, 0, improvement, diversity, stagnation, quality))
    
    # Switch at high stagnation
    trajectory.append((Phase.EXPLORATION, 1, 0, 0.3, 0.7, 0.25))  # Timely switch
    
    # Exploitation: 25 steps, no improvements
    for i in range(25):
        improvement = 0  # No luck
        stagnation = 0.3 + (i / 25) * 0.5
        quality = 0.25 + (i / 25) * 0.15  # Tiny improvement
        trajectory.append((Phase.EXPLOITATION, 0, improvement, 0.2, stagnation, quality))
    
    trajectory.append((Phase.EXPLOITATION, 1, 0, 0.15, 0.85, 0.4))
    trajectory.append((Phase.TERMINATION, 0, 0, 0.15, 0.85, 0.4))
    
    scenarios.append((
        "BALANCED_STAGNANT",
        "Proper timing but no improvements found",
        trajectory,
        "medium"
    ))
    
    # =========================================================================
    # SCENARIO 7: PREMATURE SWITCH (IMPROVING IN EXPLORATION)
    # Agent switches while still finding improvements
    # Should be PENALIZED heavily
    # =========================================================================
    trajectory = []
    
    # 10 exploration steps with consistent improvements
    for i in range(10):
        improvement = 1  # All improvements!
        diversity = 0.9 - (i / 10) * 0.2
        stagnation = 0.1
        quality = 0.1 + (i / 10) * 0.4
        trajectory.append((Phase.EXPLORATION, 0, improvement, diversity, stagnation, quality))
    
    # Switch while still improving! BAD
    trajectory.append((Phase.EXPLORATION, 1, 1, 0.7, 0.1, 0.5))  # Premature!
    
    # Exploitation
    for i in range(20):
        improvement = 1 if i < 5 else 0
        stagnation = 0.2 + (i / 20) * 0.6
        quality = 0.5 + (i / 20) * 0.2
        trajectory.append((Phase.EXPLOITATION, 0, improvement, 0.4, stagnation, quality))
    
    trajectory.append((Phase.EXPLOITATION, 1, 0, 0.3, 0.8, 0.7))
    trajectory.append((Phase.TERMINATION, 0, 0, 0.3, 0.8, 0.7))
    
    scenarios.append((
        "PREMATURE_SWITCH_IMPROVING",
        "Switch while exploration is still finding improvements",
        trajectory,
        "medium"  # Lost potential, but still decent quality
    ))
    
    # =========================================================================
    # SCENARIO 8: ULTRA FAST EXIT (3 steps total)
    # This is what our agent seems to be doing!
    # =========================================================================
    trajectory = [
        (Phase.EXPLORATION, 1, 0, 0.3, 0.1, 0.15),  # Immediate switch - very premature
        (Phase.EXPLOITATION, 1, 0, 0.2, 0.2, 0.18),  # Immediate advance - very premature
        (Phase.TERMINATION, 0, 0, 0.2, 0.2, 0.18),  # Terrible quality
    ]
    scenarios.append((
        "ULTRA_FAST_EXIT_3_STEPS",
        "Agent exits in just 3 steps (observed bug behavior)",
        trajectory,
        "low"
    ))
    
    # =========================================================================
    # SCENARIO 9: FAST EXIT (10 steps) with OK quality
    # Short episode but reasonable switching
    # =========================================================================
    trajectory = [
        # 5 exploration steps with some improvements
        (Phase.EXPLORATION, 0, 1, 0.7, 0.1, 0.3),
        (Phase.EXPLORATION, 0, 0, 0.65, 0.2, 0.32),
        (Phase.EXPLORATION, 0, 1, 0.6, 0.25, 0.38),
        (Phase.EXPLORATION, 0, 0, 0.55, 0.35, 0.4),
        (Phase.EXPLORATION, 0, 0, 0.5, 0.45, 0.42),
        # Switch at moderate stagnation
        (Phase.EXPLORATION, 1, 0, 0.5, 0.5, 0.42),
        # 3 exploitation steps
        (Phase.EXPLOITATION, 0, 1, 0.4, 0.2, 0.55),
        (Phase.EXPLOITATION, 0, 0, 0.35, 0.4, 0.58),
        (Phase.EXPLOITATION, 0, 0, 0.3, 0.6, 0.6),
        # Advance to termination
        (Phase.EXPLOITATION, 1, 0, 0.3, 0.65, 0.6),
        (Phase.TERMINATION, 0, 0, 0.3, 0.65, 0.6),
    ]
    scenarios.append((
        "FAST_EXIT_10_STEPS_GOOD",
        "Short episode (10 steps) with proper timing and OK quality",
        trajectory,
        "medium"
    ))
    
    # =========================================================================
    # SCENARIO 10: LONG EXPLOITATION HIGH QUALITY
    # Skip exploration, but achieve excellent quality through long exploitation
    # =========================================================================
    trajectory = [
        # Minimal exploration
        (Phase.EXPLORATION, 1, 0, 0.4, 0.2, 0.2),
    ]
    
    # 60 exploitation steps with steady improvement
    for i in range(60):
        improvement = 1 if i % 3 == 0 else 0  # Frequent improvements
        stagnation = min(0.8, 0.1 + i * 0.012)
        quality = min(0.95, 0.2 + i * 0.0125)  # Reaches 0.95
        trajectory.append((Phase.EXPLOITATION, 0, improvement, 0.2, stagnation, quality))
    
    trajectory.append((Phase.EXPLOITATION, 1, 0, 0.15, 0.85, 0.95))
    trajectory.append((Phase.TERMINATION, 0, 0, 0.15, 0.85, 0.95))
    
    scenarios.append((
        "LONG_EXPLOITATION_HIGH_QUALITY",
        "Skip exploration but achieve excellent quality through persistence",
        trajectory,
        "high"  # Quality matters!
    ))
    
    return scenarios


def run_validation(config: Optional[ISRConfig] = None, max_steps: int = 100, verbose: bool = True):
    """
    Run all episode simulations and validate rankings.
    """
    simulator = EpisodeSimulator(config, max_steps)
    scenarios = create_episode_scenarios(max_steps)
    
    results: List[EpisodeResult] = []
    
    print("=" * 80)
    print("EPISODE SIMULATION VALIDATOR - ISR v2.0")
    print("=" * 80)
    print(f"Max Steps per Episode: {max_steps}")
    print(f"Total Scenarios: {len(scenarios)}")
    print("=" * 80)
    print()
    
    for name, desc, trajectory, expected_rank in scenarios:
        result = simulator.run_episode(name, desc, trajectory, expected_rank)
        results.append(result)
        
        if verbose:
            print(result.summary())
    
    print()
    print("-" * 80)
    print("RANKING VALIDATION")
    print("-" * 80)
    
    # Sort by total reward
    sorted_results = sorted(results, key=lambda x: x.total_reward, reverse=True)
    
    # Classify into groups
    high_expected = [r for r in results if r.expected_ranking == "high"]
    medium_expected = [r for r in results if r.expected_ranking == "medium"]
    low_expected = [r for r in results if r.expected_ranking == "low"]
    
    # Compute average rewards per group
    avg_high = np.mean([r.total_reward for r in high_expected]) if high_expected else 0
    avg_medium = np.mean([r.total_reward for r in medium_expected]) if medium_expected else 0
    avg_low = np.mean([r.total_reward for r in low_expected]) if low_expected else 0
    
    print(f"\nExpected Group Averages:")
    print(f"  HIGH expected:   avg={avg_high:+.3f}  ({len(high_expected)} scenarios)")
    print(f"  MEDIUM expected: avg={avg_medium:+.3f}  ({len(medium_expected)} scenarios)")
    print(f"  LOW expected:    avg={avg_low:+.3f}  ({len(low_expected)} scenarios)")
    
    # Validation checks
    errors = []
    
    # Check 1: HIGH > MEDIUM > LOW in average
    if not (avg_high > avg_medium):
        errors.append(f"HIGH ({avg_high:.3f}) should be > MEDIUM ({avg_medium:.3f})")
    if not (avg_medium > avg_low):
        errors.append(f"MEDIUM ({avg_medium:.3f}) should be > LOW ({avg_low:.3f})")
    
    # Check 2: No LOW scenario should beat any HIGH scenario
    if high_expected and low_expected:
        min_high = min(r.total_reward for r in high_expected)
        max_low = max(r.total_reward for r in low_expected)
        if max_low > min_high:
            # Find the offending scenarios
            for low_r in low_expected:
                for high_r in high_expected:
                    if low_r.total_reward > high_r.total_reward:
                        errors.append(f"LOW scenario '{low_r.name}' ({low_r.total_reward:.3f}) beats HIGH '{high_r.name}' ({high_r.total_reward:.3f})")
    
    print()
    print("-" * 80)
    print("DETAILED RANKINGS (Sorted by Total Reward)")
    print("-" * 80)
    
    for i, r in enumerate(sorted_results):
        marker = "✓" if r.expected_ranking == "high" and i < len(high_expected) else ""
        marker = marker or ("✓" if r.expected_ranking == "low" and i >= len(results) - len(low_expected) else "")
        marker = marker or "⚠" if not marker else marker
        
        print(f"{i+1:2d}. [{r.expected_ranking.upper():6s}] {marker} {r.name[:35]:35s} "
              f"Total={r.total_reward:+.3f}  Q={r.final_quality:.2f}  "
              f"E={r.explore_steps:2d}  X={r.exploit_steps:2d}")
    
    print()
    print("=" * 80)
    print("VALIDATION RESULT")
    print("=" * 80)
    
    if errors:
        print("❌ VALIDATION FAILED - Reward function has ranking issues:")
        for e in errors:
            print(f"  - {e}")
        print()
        print("This explains the agent behavior - bad strategies are being rewarded!")
        return False, results, errors
    else:
        print("✓ All ranking validations passed")
        print("  HIGH strategies correctly rewarded more than MEDIUM > LOW")
        return True, results, []


def detailed_episode_breakdown(result: EpisodeResult):
    """Print detailed step-by-step breakdown of an episode."""
    print(f"\n{'='*80}")
    print(f"DETAILED BREAKDOWN: {result.name}")
    print(f"{'='*80}")
    print(f"Description: {result.description}")
    print(f"Expected Ranking: {result.expected_ranking.upper()}")
    print(f"Total Steps: {len(result.steps)}")
    print(f"Explore Steps: {result.explore_steps}")
    print(f"Exploit Steps: {result.exploit_steps}")
    print(f"Final Quality: {result.final_quality:.3f}")
    print(f"TOTAL REWARD: {result.total_reward:+.4f}")
    print()
    
    print(f"{'Step':>4} {'Phase':>12} {'Act':>3} {'Imp':>3} {'Div':>5} {'Stag':>5} {'Qual':>5} {'Budg':>5} {'Reward':>8} Components")
    print("-" * 100)
    
    cumulative = 0.0
    for s in result.steps:
        cumulative += s.reward
        comp_str = ", ".join(f"{k}={v:+.3f}" for k, v in s.components.items())
        print(f"{s.step:4d} {s.phase.value:>12} {s.action:3d} {s.improvement:3d} "
              f"{s.diversity:5.2f} {s.stagnation:5.2f} {s.quality:5.2f} {s.budget:5.2f} "
              f"{s.reward:+8.4f} [{comp_str}]")
    
    print("-" * 100)
    print(f"{'CUMULATIVE':>52} {cumulative:+8.4f}")


def main():
    """Main validation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Episode Simulation Validator for ISR")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--breakdown", type=str, help="Show detailed breakdown for scenario name")
    
    args = parser.parse_args()
    
    passed, results, errors = run_validation(
        config=None,  # Use default config
        max_steps=args.max_steps,
        verbose=True
    )
    
    # Show breakdown for specific scenario if requested
    if args.breakdown:
        for r in results:
            if args.breakdown.lower() in r.name.lower():
                detailed_episode_breakdown(r)
                break
    
    # Always show breakdown for the bug scenarios
    print("\n" + "=" * 80)
    print("CRITICAL ANALYSIS: Why is Early Exit Being Learned?")
    print("=" * 80)
    
    # Find the early exit scenarios
    early_exit = None
    balanced_good = None
    ultra_fast = None
    
    for r in results:
        if "ULTRA_FAST" in r.name:
            ultra_fast = r
        elif "EARLY_EXIT_LOW" in r.name:
            early_exit = r
        elif "BALANCED_GOOD" in r.name:
            balanced_good = r
    
    if ultra_fast:
        detailed_episode_breakdown(ultra_fast)
    if early_exit:
        detailed_episode_breakdown(early_exit)
    if balanced_good:
        detailed_episode_breakdown(balanced_good)
    
    # Final diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if not passed:
        print("\n⚠️  REWARD FUNCTION HAS ISSUES:")
        for e in errors:
            print(f"  - {e}")
    
    if ultra_fast and balanced_good:
        reward_ratio = balanced_good.total_reward / (ultra_fast.total_reward + 1e-8) if ultra_fast.total_reward != 0 else float('inf')
        steps_ratio = len(balanced_good.steps) / len(ultra_fast.steps)
        
        print(f"\nComparison: BALANCED_GOOD vs ULTRA_FAST_EXIT")
        print(f"  BALANCED_GOOD:  {balanced_good.total_reward:+.3f} in {len(balanced_good.steps)} steps")
        print(f"  ULTRA_FAST:     {ultra_fast.total_reward:+.3f} in {len(ultra_fast.steps)} steps")
        print(f"  Reward Ratio:   {reward_ratio:.2f}x better")
        print(f"  Steps Ratio:    {steps_ratio:.1f}x more steps")
        
        # Per-step reward comparison
        per_step_balanced = balanced_good.total_reward / len(balanced_good.steps)
        per_step_ultra = ultra_fast.total_reward / len(ultra_fast.steps)
        
        print(f"\n  Per-Step Reward:")
        print(f"    BALANCED_GOOD: {per_step_balanced:+.4f} per step")
        print(f"    ULTRA_FAST:    {per_step_ultra:+.4f} per step")
        
        if per_step_ultra > per_step_balanced:
            print("\n  ⚠️  PROBLEM FOUND: Per-step reward is HIGHER for early exit!")
            print("      This creates incentive to exit quickly.")
        
        if ultra_fast.total_reward > 0:
            print("\n  ⚠️  PROBLEM: Ultra-fast exit has POSITIVE total reward!")
            print("      Early exit should be clearly negative.")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
