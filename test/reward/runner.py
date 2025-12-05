import json
import sys
import os
import numpy as np  # Added for linspace

# Ensure we can import from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RLOrchestrator.core.reward import EffectivenessFirstReward

def load_scenarios(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_validation():
    base_dir = os.path.dirname(__file__)
    scenarios_path = os.path.join(base_dir, 'scenarios.json')
    
    print(f"Loading scenarios from {scenarios_path}...")
    try:
        scenarios = load_scenarios(scenarios_path)
    except FileNotFoundError:
        print("Error: scenarios.json not found.")
        sys.exit(1)

    reward_computer = EffectivenessFirstReward()
    
    passed = 0
    failed = 0
    
    print("\n" + "="*100)
    print(f"{'SCENARIO ID':<25} | {'RESULT':<10} | {'REWARD':<8} | {'NOTES'}")
    print("="*100)

    for scenario in scenarios["single_state_scenarios"]:
        # Skip section headers
        if "_section" in scenario:
            continue
        s_id = scenario['id']
        phase = scenario['phase']
        inputs = scenario['inputs']
        action = scenario['action']
        assertions = scenario['assertions']
        
        try:
            signal = reward_computer.calculate(inputs, action, phase)
            val = signal.value
            components = signal.components
            
            errors = []
            
            # Check Min Value
            if 'min_value' in assertions:
                if val < assertions['min_value']:
                    errors.append(f"Too Low ({val:.2f} < {assertions['min_value']})")
            
            # Check Max Value
            if 'max_value' in assertions:
                if val > assertions['max_value']:
                    errors.append(f"Too High ({val:.2f} > {assertions['max_value']})")
            
            # Check Components
            if 'must_have_component' in assertions:
                req_comp = assertions['must_have_component']
                if req_comp not in components:
                    errors.append(f"Missing '{req_comp}'")
                elif components[req_comp] == 0:
                    errors.append(f"Zero '{req_comp}'")

            if not errors:
                # Format components for display
                comp_str = ", ".join([f"{k}:{v:.2f}" for k,v in components.items()])
                print(f"{s_id:<25} | {'PASS':<10} | {val:>6.2f}   | {comp_str}")
                passed += 1
            else:
                print(f"{s_id:<25} | {'FAIL':<10} | {val:>6.2f}   | {', '.join(errors)}")
                failed += 1

        except Exception as e:
            print(f"{s_id:<25} | {'ERROR':<10} | {'N/A':>6}   | {str(e)}")
            failed += 1

    print("="*100)
    print(f"SCENARIO SUMMARY: {passed} Passed, {failed} Failed")
    
    # Run the new Gradient Tests
    gradient_failures = run_gradient_analysis(reward_computer)
    
    total_failures = failed + gradient_failures
    if total_failures > 0:
        sys.exit(1)

def run_gradient_analysis(reward_computer):
    """
    Performs continuous parameter sweeps to detect 'Cliffs' (Discontinuities)
    that would cause exploding gradients in DRL training.
    """
    print("\n" + "="*100)
    print("GRADIENT & SMOOTHNESS ANALYSIS (ML Compatibility)")
    print("="*100)
    print(f"{'PARAMETER SWEEP':<30} | {'RESULT':<10} | {'MAX JUMP':<10} | {'COMMENT'}")
    print("-" * 100)

    failures = 0
    
    # Threshold for a "Cliff" (Sudden jump in reward value)
    # If reward changes by more than 0.3 in a 0.01 step, that's a cliff.
    CLIFF_THRESHOLD = 0.3 

    # --- TEST 1: Stagnation Smoothness (Exploration) ---
    # Context: Exploration, No Improvement. Stagnation increases 0 -> 1.
    # Expectation: Reward should decrease smoothly.
    max_delta = 0.0
    prev_val = None
    
    inputs = {
        "fitness_norm": 0.5,
        "improvement_velocity": 0.0,
        "diversity_score": 0.5,
        "budget_consumed": 0.1
    }
    phase = 'EXPLORATION'
    
    for stag in np.linspace(0, 1, 100):
        inputs['stagnation_ratio'] = stag
        signal = reward_computer.calculate(inputs, 0, phase)  # Action: Continue
        val = signal.value
        
        if prev_val is not None:
            delta = abs(val - prev_val)
            max_delta = max(max_delta, delta)
        prev_val = val

    status = "PASS" if max_delta < CLIFF_THRESHOLD else "FAIL"
    if status == "FAIL": failures += 1
    print(f"{'Stagnation (Exploration)':<30} | {status:<10} | {max_delta:>6.4f}     | {'Smooth penalty' if status=='PASS' else 'Detected Cliff!'}")

    # --- TEST 2: Fitness Monotonicity (Exploitation) ---
    # Context: Exploitation, Improvement. Fitness increases 0 -> 1.
    # Expectation: Reward should increase smoothly.
    max_delta = 0.0
    prev_val = None
    
    inputs = {
        "stagnation_ratio": 0.0,
        "improvement_velocity": 0.8,
        "diversity_score": 0.1,
        "budget_consumed": 0.1
    }
    phase = 'EXPLOITATION'
    
    for fit in np.linspace(0, 1, 100):
        inputs['fitness_norm'] = fit
        signal = reward_computer.calculate(inputs, 0, phase)  # Action: Continue
        val = signal.value
        
        if prev_val is not None:
            delta = abs(val - prev_val)
            max_delta = max(max_delta, delta)
        prev_val = val

    status = "PASS" if max_delta < CLIFF_THRESHOLD else "FAIL"
    if status == "FAIL": failures += 1
    print(f"{'Fitness (Exploitation)':<30} | {status:<10} | {max_delta:>6.4f}     | {'Smooth scaling' if status=='PASS' else 'Detected Cliff!'}")

    # --- TEST 3: Budget Pressure (Global) ---
    # Context: Exploration. Budget increases 0.8 -> 1.0.
    # Expectation: Reward for Continuing should drop smoothly, not crash at 0.95.
    max_delta = 0.0
    prev_val = None
    
    inputs = {
        "fitness_norm": 0.5,
        "improvement_velocity": 0.0,
        "diversity_score": 0.5,
        "stagnation_ratio": 0.0
    }
    phase = 'EXPLORATION'
    
    for budget in np.linspace(0.8, 1.0, 50):
        inputs['budget_consumed'] = budget
        signal = reward_computer.calculate(inputs, 0, phase)  # Action: Continue
        val = signal.value
        
        if prev_val is not None:
            delta = abs(val - prev_val)
            max_delta = max(max_delta, delta)
        prev_val = val

    status = "PASS" if max_delta < CLIFF_THRESHOLD else "FAIL"
    if status == "FAIL": failures += 1
    print(f"{'Budget Pressure':<30} | {status:<10} | {max_delta:>6.4f}     | {'Smooth pressure' if status=='PASS' else 'Detected Cliff!'}")

    # --- TEST 4: Monte Carlo Lipschitz Stress Test (Multivariate) ---
    # Samples random points in 5D space to find hidden interaction cliffs.
    print("-" * 100)
    print(f"{'MONTE CARLO STRESS TEST':<30} | {'RESULT':<10} | {'MAX K':<10} | {'COMMENT'}")
    
    max_k = 0.0
    worst_case = None
    phase = 'EXPLORATION'
    
    # Sample 1000 random states
    np.random.seed(42)
    for _ in range(1000):
        # Generate random state [0, 1]
        base_state = {
            "fitness_norm": np.random.random(),
            "improvement_velocity": np.random.random(),
            "diversity_score": np.random.random(),
            "stagnation_ratio": np.random.random(),
            "budget_consumed": np.random.random()
        }
        
        # Create a tiny perturbation
        epsilon = 0.001
        perturbed_state = base_state.copy()
        # Perturb one continuous variable randomly
        key = np.random.choice(["fitness_norm", "diversity_score", "stagnation_ratio", "budget_consumed", "improvement_velocity"])
        perturbed_state[key] = np.clip(base_state[key] + epsilon, 0.0, 1.0)
        
        # Calculate actual distance (might be less than epsilon due to clipping)
        dist = abs(perturbed_state[key] - base_state[key])
        if dist < 1e-9: continue
            
        # Get rewards
        r1 = reward_computer.calculate(base_state, 0, phase).value
        r2 = reward_computer.calculate(perturbed_state, 0, phase).value
        
        # Calculate Lipschitz Constant K (Rate of Change)
        k = abs(r1 - r2) / dist
        
        if k > max_k:
            max_k = k
            worst_case = f"Jump at {key}={base_state[key]:.2f}"

    # K > 10.0 implies a slope so steep it's effectively a cliff for the optimizer
    LIPSCHITZ_LIMIT = 10.0
    status = "PASS" if max_k < LIPSCHITZ_LIMIT else "FAIL"
    if status == "FAIL": failures += 1
    
    print(f"{'Global Lipschitz Const':<30} | {status:<10} | {max_k:>6.2f}     | {worst_case if status=='FAIL' else 'No explosions detected'}")

    return failures

if __name__ == "__main__":
    run_validation()
