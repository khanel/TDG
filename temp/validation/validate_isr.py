#!/usr/bin/env python3
"""
Comprehensive Validation Runner for Intelligent Search Reward (ISR)

This script validates the reward function against:
1. Behavioral scenarios (correct rewards for specific situations)
2. Mathematical properties (smoothness, Lipschitz continuity, boundedness)
3. Gradient analysis (no cliffs that would destabilize training)

Usage:
    python validate_isr.py                    # Run all tests
    python validate_isr.py --scenarios-only   # Only behavioral tests
    python validate_isr.py --math-only        # Only mathematical tests
"""

import json
import sys
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Ensure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from temp.core.intelligent_reward import IntelligentSearchReward, ISRConfig


@dataclass
class ValidationResult:
    passed: int
    failed: int
    errors: List[str]


def load_scenarios(filepath: str) -> List[Dict]:
    """Load test scenarios from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_behavioral_validation(reward_computer: IntelligentSearchReward, scenarios: List[Dict]) -> ValidationResult:
    """
    Run behavioral validation against predefined scenarios.
    Each scenario tests a specific expected behavior.
    """
    print("\n" + "=" * 110)
    print("BEHAVIORAL VALIDATION (Scenario-Based)")
    print("=" * 110)
    print(f"{'SCENARIO ID':<35} | {'RESULT':<8} | {'REWARD':>8} | {'NOTES'}")
    print("-" * 110)
    
    passed = 0
    failed = 0
    errors = []
    
    for scenario in scenarios:
        s_id = scenario['id']
        phase = scenario['phase']
        inputs = scenario['inputs']
        action = scenario['action']
        assertions = scenario['assertions']
        
        context = {'current_phase': phase}
        
        try:
            signal = reward_computer.calculate(inputs, action, context)
            val = signal.total_value
            components = signal.raw_components
            
            scenario_errors = []
            
            # Check Min Value
            if 'min_value' in assertions:
                if val < assertions['min_value']:
                    scenario_errors.append(f"Too Low ({val:.3f} < {assertions['min_value']})")
            
            # Check Max Value
            if 'max_value' in assertions:
                if val > assertions['max_value']:
                    scenario_errors.append(f"Too High ({val:.3f} > {assertions['max_value']})")
            
            # Check Required Component
            if 'must_have_component' in assertions:
                req_comp = assertions['must_have_component']
                if req_comp not in components:
                    scenario_errors.append(f"Missing '{req_comp}'")
                elif abs(components[req_comp]) < 1e-9:
                    scenario_errors.append(f"Zero '{req_comp}'")
            
            if not scenario_errors:
                comp_str = ", ".join([f"{k}:{v:.3f}" for k, v in components.items()])
                print(f"{s_id:<35} | {'PASS':<8} | {val:>8.3f} | {comp_str[:50]}")
                passed += 1
            else:
                error_str = ", ".join(scenario_errors)
                print(f"{s_id:<35} | {'FAIL':<8} | {val:>8.3f} | {error_str}")
                failed += 1
                errors.append(f"{s_id}: {error_str}")
        
        except Exception as e:
            print(f"{s_id:<35} | {'ERROR':<8} | {'N/A':>8} | {str(e)}")
            failed += 1
            errors.append(f"{s_id}: Exception - {str(e)}")
    
    print("-" * 110)
    print(f"BEHAVIORAL SUMMARY: {passed} Passed, {failed} Failed")
    
    return ValidationResult(passed=passed, failed=failed, errors=errors)


def run_smoothness_analysis(reward_computer: IntelligentSearchReward) -> ValidationResult:
    """
    Perform continuous parameter sweeps to detect discontinuities (cliffs).
    A cliff would cause exploding gradients in DRL training.
    """
    print("\n" + "=" * 110)
    print("SMOOTHNESS ANALYSIS (Gradient Stability)")
    print("=" * 110)
    print(f"{'PARAMETER SWEEP':<35} | {'RESULT':<8} | {'MAX JUMP':>10} | {'COMMENT'}")
    print("-" * 110)
    
    passed = 0
    failed = 0
    errors = []
    
    # Threshold for detecting a cliff
    CLIFF_THRESHOLD = 0.3
    
    # --- TEST 1: Stagnation Smoothness (Exploration, Continue) ---
    max_delta = 0.0
    prev_val = None
    
    inputs = {
        "fitness_norm": 0.5,
        "improvement_binary": 0,
        "diversity_score": 0.5,
        "budget_consumed": 0.3,
        "stagnation_ratio": 0.0
    }
    context = {'current_phase': 'EXPLORATION'}
    
    for stag in np.linspace(0, 1, 100):
        inputs['stagnation_ratio'] = stag
        signal = reward_computer.calculate(inputs, 0, context)
        val = signal.total_value
        
        if prev_val is not None:
            delta = abs(val - prev_val)
            max_delta = max(max_delta, delta)
        prev_val = val
    
    status = "PASS" if max_delta < CLIFF_THRESHOLD else "FAIL"
    if status == "FAIL":
        failed += 1
        errors.append(f"Stagnation (Exploration): Cliff detected, max_delta={max_delta:.4f}")
    else:
        passed += 1
    print(f"{'Stagnation (Exploration, Stay)':<35} | {status:<8} | {max_delta:>10.4f} | {'Smooth penalty ramp' if status == 'PASS' else 'CLIFF DETECTED!'}")
    
    # --- TEST 2: Fitness Monotonicity (Exploitation, Continue) ---
    max_delta = 0.0
    prev_val = None
    
    inputs = {
        "stagnation_ratio": 0.0,
        "improvement_binary": 1,
        "diversity_score": 0.1,
        "budget_consumed": 0.3,
        "fitness_norm": 0.0
    }
    context = {'current_phase': 'EXPLOITATION'}
    
    for fit in np.linspace(0, 1, 100):
        inputs['fitness_norm'] = fit
        signal = reward_computer.calculate(inputs, 0, context)
        val = signal.total_value
        
        if prev_val is not None:
            delta = abs(val - prev_val)
            max_delta = max(max_delta, delta)
        prev_val = val
    
    status = "PASS" if max_delta < CLIFF_THRESHOLD else "FAIL"
    if status == "FAIL":
        failed += 1
        errors.append(f"Fitness (Exploitation): Cliff detected, max_delta={max_delta:.4f}")
    else:
        passed += 1
    print(f"{'Fitness (Exploitation, Continue)':<35} | {status:<8} | {max_delta:>10.4f} | {'Smooth quality scaling' if status == 'PASS' else 'CLIFF DETECTED!'}")
    
    # --- TEST 3: Budget Pressure Smoothness ---
    max_delta = 0.0
    prev_val = None
    
    inputs = {
        "fitness_norm": 0.5,
        "improvement_binary": 0,
        "diversity_score": 0.5,
        "stagnation_ratio": 0.3,
        "budget_consumed": 0.0
    }
    context = {'current_phase': 'EXPLORATION'}
    
    for budget in np.linspace(0.5, 1.0, 100):
        inputs['budget_consumed'] = budget
        signal = reward_computer.calculate(inputs, 0, context)
        val = signal.total_value
        
        if prev_val is not None:
            delta = abs(val - prev_val)
            max_delta = max(max_delta, delta)
        prev_val = val
    
    status = "PASS" if max_delta < CLIFF_THRESHOLD else "FAIL"
    if status == "FAIL":
        failed += 1
        errors.append(f"Budget Pressure: Cliff detected, max_delta={max_delta:.4f}")
    else:
        passed += 1
    print(f"{'Budget Pressure (0.5 -> 1.0)':<35} | {status:<8} | {max_delta:>10.4f} | {'Smooth pressure ramp' if status == 'PASS' else 'CLIFF DETECTED!'}")
    
    # --- TEST 4: Diversity Smoothness (Exploration) ---
    max_delta = 0.0
    prev_val = None
    
    inputs = {
        "fitness_norm": 0.5,
        "improvement_binary": 1,  # With improvement
        "stagnation_ratio": 0.1,
        "budget_consumed": 0.3,
        "diversity_score": 0.0
    }
    context = {'current_phase': 'EXPLORATION'}
    
    for div in np.linspace(0, 1, 100):
        inputs['diversity_score'] = div
        signal = reward_computer.calculate(inputs, 0, context)
        val = signal.total_value
        
        if prev_val is not None:
            delta = abs(val - prev_val)
            max_delta = max(max_delta, delta)
        prev_val = val
    
    status = "PASS" if max_delta < CLIFF_THRESHOLD else "FAIL"
    if status == "FAIL":
        failed += 1
        errors.append(f"Diversity (Exploration): Cliff detected, max_delta={max_delta:.4f}")
    else:
        passed += 1
    print(f"{'Diversity (Exploration, Improve)':<35} | {status:<8} | {max_delta:>10.4f} | {'Smooth diversity bonus' if status == 'PASS' else 'CLIFF DETECTED!'}")
    
    print("-" * 110)
    print(f"SMOOTHNESS SUMMARY: {passed} Passed, {failed} Failed")
    
    return ValidationResult(passed=passed, failed=failed, errors=errors)


def run_lipschitz_analysis(reward_computer: IntelligentSearchReward, num_samples: int = 2000) -> ValidationResult:
    """
    Monte Carlo stress test to estimate global Lipschitz constant.
    K > 10 indicates potential training instability.
    """
    print("\n" + "=" * 110)
    print("LIPSCHITZ CONTINUITY ANALYSIS (Monte Carlo)")
    print("=" * 110)
    
    passed = 0
    failed = 0
    errors = []
    
    LIPSCHITZ_LIMIT = 10.0
    
    np.random.seed(42)
    
    max_k = 0.0
    worst_case_info = ""
    
    # Test all 3 phases
    for phase in ['EXPLORATION', 'EXPLOITATION', 'TERMINATION']:
        context = {'current_phase': phase}
        
        # Termination phase has no action
        actions = [0] if phase == 'TERMINATION' else [0, 1]
        
        for action in actions:
            phase_max_k = 0.0
            phase_worst = ""
            
            for _ in range(num_samples):
                # Generate random base state
                base_state = {
                    "fitness_norm": np.random.random(),
                    "improvement_binary": np.random.choice([0, 1]),
                    "diversity_score": np.random.random(),
                    "stagnation_ratio": np.random.random(),
                    "budget_consumed": np.random.random()
                }
                
                # Perturb one continuous variable
                epsilon = 0.001
                perturbed_state = base_state.copy()
                key = np.random.choice(["fitness_norm", "diversity_score", "stagnation_ratio", "budget_consumed"])
                perturbed_state[key] = np.clip(base_state[key] + epsilon, 0.0, 1.0)
                
                # Calculate distance
                dist = abs(perturbed_state[key] - base_state[key])
                if dist < 1e-9:
                    continue
                
                # Get rewards
                r1 = reward_computer.calculate(base_state, action, context).total_value
                r2 = reward_computer.calculate(perturbed_state, action, context).total_value
                
                # Calculate Lipschitz constant
                k = abs(r1 - r2) / dist
                
                if k > phase_max_k:
                    phase_max_k = k
                    phase_worst = f"{phase}/action={action}/{key}={base_state[key]:.2f}"
            
            if phase_max_k > max_k:
                max_k = phase_max_k
                worst_case_info = phase_worst
    
    status = "PASS" if max_k < LIPSCHITZ_LIMIT else "FAIL"
    if status == "FAIL":
        failed += 1
        errors.append(f"Lipschitz constant {max_k:.2f} exceeds limit {LIPSCHITZ_LIMIT}")
    else:
        passed += 1
    
    print(f"{'TEST':<35} | {'RESULT':<8} | {'MAX K':>10} | {'WORST CASE'}")
    print("-" * 110)
    print(f"{'Global Lipschitz Constant':<35} | {status:<8} | {max_k:>10.2f} | {worst_case_info if status == 'FAIL' else 'No explosions detected'}")
    print("-" * 110)
    print(f"LIPSCHITZ SUMMARY: {passed} Passed, {failed} Failed")
    print(f"(Limit: K < {LIPSCHITZ_LIMIT}, Measured: K = {max_k:.2f})")
    
    return ValidationResult(passed=passed, failed=failed, errors=errors)


def run_boundedness_check(reward_computer: IntelligentSearchReward, num_samples: int = 5000) -> ValidationResult:
    """
    Verify all rewards are within [-1, 1] bounds via exhaustive sampling.
    """
    print("\n" + "=" * 110)
    print("BOUNDEDNESS VERIFICATION")
    print("=" * 110)
    
    passed = 0
    failed = 0
    errors = []
    
    np.random.seed(123)
    
    min_reward = float('inf')
    max_reward = float('-inf')
    violations = 0
    
    # Test all 3 phases
    for phase in ['EXPLORATION', 'EXPLOITATION', 'TERMINATION']:
        context = {'current_phase': phase}
        
        # Termination phase has no action (episode ends)
        actions = [0] if phase == 'TERMINATION' else [0, 1]
        
        for action in actions:
            for _ in range(num_samples):
                state = {
                    "fitness_norm": np.random.random(),
                    "improvement_binary": np.random.choice([0, 1]),
                    "diversity_score": np.random.random(),
                    "stagnation_ratio": np.random.random(),
                    "budget_consumed": np.random.random()
                }
                
                signal = reward_computer.calculate(state, action, context)
                val = signal.total_value
                
                min_reward = min(min_reward, val)
                max_reward = max(max_reward, val)
                
                if val < -1.0 - 1e-6 or val > 1.0 + 1e-6:
                    violations += 1
    
    status = "PASS" if violations == 0 else "FAIL"
    if status == "FAIL":
        failed += 1
        errors.append(f"Found {violations} out-of-bounds rewards")
    else:
        passed += 1
    
    print(f"{'TEST':<35} | {'RESULT':<8} | {'RANGE':>20} | {'VIOLATIONS'}")
    print("-" * 110)
    print(f"{'Reward Boundedness [-1, 1]':<35} | {status:<8} | [{min_reward:>7.3f}, {max_reward:>7.3f}] | {violations}")
    print("-" * 110)
    print(f"BOUNDEDNESS SUMMARY: {passed} Passed, {failed} Failed")
    
    return ValidationResult(passed=passed, failed=failed, errors=errors)


def print_final_report(results: Dict[str, ValidationResult]):
    """Print comprehensive final report."""
    print("\n" + "=" * 110)
    print("FINAL VALIDATION REPORT")
    print("=" * 110)
    
    total_passed = sum(r.passed for r in results.values())
    total_failed = sum(r.failed for r in results.values())
    
    print(f"\n{'Category':<30} | {'Passed':>8} | {'Failed':>8} | {'Status'}")
    print("-" * 60)
    
    for category, result in results.items():
        status = "✅ OK" if result.failed == 0 else "❌ FAIL"
        print(f"{category:<30} | {result.passed:>8} | {result.failed:>8} | {status}")
    
    print("-" * 60)
    print(f"{'TOTAL':<30} | {total_passed:>8} | {total_failed:>8} | {'✅ ALL TESTS PASSED' if total_failed == 0 else '❌ SOME TESTS FAILED'}")
    
    if total_failed > 0:
        print("\n" + "=" * 110)
        print("FAILURE DETAILS")
        print("=" * 110)
        for category, result in results.items():
            if result.errors:
                print(f"\n[{category}]")
                for err in result.errors:
                    print(f"  • {err}")
    
    print("\n" + "=" * 110)
    
    if total_failed == 0:
        print("🎉 VALIDATION COMPLETE: Reward function is READY FOR TRAINING")
    else:
        print("⚠️  VALIDATION INCOMPLETE: Please fix the failing tests before training")
    
    print("=" * 110)
    
    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Validate Intelligent Search Reward Function")
    parser.add_argument("--scenarios-only", action="store_true", help="Run only behavioral scenarios")
    parser.add_argument("--math-only", action="store_true", help="Run only mathematical analysis")
    parser.add_argument("--scenarios-file", type=str, default=None, help="Path to scenarios JSON file")
    args = parser.parse_args()
    
    # Initialize reward computer
    reward_computer = IntelligentSearchReward()
    
    results = {}
    
    # Determine scenarios file path
    base_dir = os.path.dirname(__file__)
    if args.scenarios_file:
        scenarios_path = args.scenarios_file
    else:
        # Try v2 first, then fall back to original
        scenarios_path = os.path.join(base_dir, 'scenarios_v2.json')
        if not os.path.exists(scenarios_path):
            scenarios_path = os.path.join(base_dir, 'scenarios.json')
    
    if not args.math_only:
        # Load and run behavioral scenarios
        print(f"\n📋 Loading scenarios from: {scenarios_path}")
        try:
            scenarios = load_scenarios(scenarios_path)
            results['Behavioral Scenarios'] = run_behavioral_validation(reward_computer, scenarios)
        except FileNotFoundError:
            print(f"⚠️  Scenarios file not found: {scenarios_path}")
            results['Behavioral Scenarios'] = ValidationResult(passed=0, failed=1, errors=["Scenarios file not found"])
    
    if not args.scenarios_only:
        # Run mathematical analysis
        results['Smoothness Analysis'] = run_smoothness_analysis(reward_computer)
        results['Lipschitz Continuity'] = run_lipschitz_analysis(reward_computer)
        results['Boundedness Check'] = run_boundedness_check(reward_computer)
    
    # Print final report
    success = print_final_report(results)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
