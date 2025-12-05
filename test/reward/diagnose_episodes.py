#!/usr/bin/env python3
"""
Episode Trace Diagnostic - Find why agent exits early
======================================================

This script runs actual episodes and traces every step to see what the agent
actually observes and what rewards it receives.

Hypothesis: The stagnation signal is increasing too fast, or improvement
detection is broken, causing the agent to see "switch/advance is optimal"
when it actually isn't.
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from RLOrchestrator.core.orchestrator import OrchestratorEnv
from RLOrchestrator.nkl.adapter import NKLAdapter
from RLOrchestrator.nkl.solvers import NKLGWOExplorer, NKLGAExploiter
from RLOrchestrator.core.reward import RewardWrapper, EFRConfig


class TracingOrchestratorEnv(OrchestratorEnv):
    """
    OrchestratorEnv with full tracing of observations, rewards, and internal state.
    """
    def __init__(self, n_items: int = 100, k_interactions: int = 5, 
                 population_size: int = 20, **kwargs):
        problem = NKLAdapter(n_items=n_items, k_interactions=k_interactions)
        explorer = NKLGWOExplorer(problem, population_size=population_size)
        exploiter = NKLGAExploiter(problem, population_size=population_size)

        super().__init__(
            problem=problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            **kwargs,
        )
        
        self.reward_comp = RewardWrapper(EFRConfig())
        self.previous_best_fitness = float('inf')
        self.trace = []
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.previous_best_fitness = float(obs[1])
        self.trace = []
        self.step_count = 0
        
        # Record initial state
        self.trace.append({
            'step': 0,
            'type': 'RESET',
            'obs': obs.tolist(),
            'phase_idx': self._context.phase_index,
            'best_fitness_raw': self._context.best_solution.fitness if self._context.best_solution else None,
            'max_decision_steps': self._context.max_decision_steps,
        })
        return obs, info

    def step(self, action: int):
        self.step_count += 1
        
        # Record pre-step state
        phase_before = self._context.phase_index
        phase_name_before = self._context.current_phase()
        best_before = self._context.best_solution.fitness if self._context.best_solution else None
        
        # Execute step
        obs, _, terminated, truncated, info = super().step(action)
        
        # Extract observation components
        budget_consumed, fitness_norm, improvement_velocity, stagnation, diversity, active_phase = obs
        
        # Get raw fitness for debugging
        best_after = self._context.best_solution.fitness if self._context.best_solution else None
        
        # Calculate improvement (same as training.py)
        improvement = 1 if fitness_norm < self.previous_best_fitness else 0
        
        # Map active_phase encoding to phase name
        if active_phase < 0.25:
            phase_name = 'EXPLORATION'
        elif active_phase < 0.75:
            phase_name = 'EXPLOITATION'
        else:
            phase_name = 'TERMINATION'
        
        # Construct state vector for reward computation
        obs_dict = {
            'improvement_velocity': float(improvement_velocity),
            'diversity_score': diversity,
            'stagnation_ratio': stagnation,
            'budget_consumed': budget_consumed,
            'fitness_norm': fitness_norm
        }
        
        # Calculate reward using EFR
        reward_signal = self.reward_comp.compute(obs_dict, action, info)
        reward = reward_signal.value
        
        # Record trace
        self.trace.append({
            'step': self.step_count,
            'action': action,
            'action_name': 'STAY' if action == 0 else 'ADVANCE',
            'phase_before': phase_name_before,
            'phase_after': self._context.current_phase(),
            'switched': info.get('switched', False),
            'obs': {
                'budget_consumed': float(budget_consumed),
                'norm_fitness': float(fitness_norm),
                'stagnation': float(stagnation),
                'diversity': float(diversity),
                'active_phase': float(active_phase),
            },
            'improvement_detected': improvement,
            'prev_norm_fitness': float(self.previous_best_fitness),
            'raw_fitness_before': best_before,
            'raw_fitness_after': best_after,
            'reward': float(reward),
            'reward_components': reward_signal.components,
            'terminated': terminated,
            'truncated': truncated,
        })
        
        self.previous_best_fitness = fitness_norm
        
        return obs, reward, terminated, truncated, info

    def print_trace(self, last_n: int = None):
        """Print the trace for analysis."""
        trace = self.trace if last_n is None else self.trace[-last_n:]
        
        print("\n" + "=" * 100)
        print("EPISODE TRACE")
        print("=" * 100)
        
        total_reward = 0
        for entry in trace:
            if entry['type'] == 'RESET' if 'type' in entry else False:
                print(f"\n{'='*80}")
                print(f"RESET - Max Decision Steps: {entry.get('max_decision_steps', '?')}")
                print(f"Initial Obs: {entry['obs']}")
                print(f"Best Fitness (raw): {entry.get('best_fitness_raw', 'N/A')}")
                continue
            
            step = entry['step']
            action = entry['action_name']
            phase_before = entry['phase_before']
            phase_after = entry['phase_after']
            obs = entry['obs']
            reward = entry['reward']
            components = entry['reward_components']
            improvement = entry['improvement_detected']
            terminated = entry['terminated']
            truncated = entry['truncated']
            
            total_reward += reward
            
            # Highlight key events
            switched = "🔄 SWITCHED" if entry.get('switched') else ""
            ended = "🏁 TERMINATED" if terminated else ("⏰ TRUNCATED" if truncated else "")
            
            print(f"\n--- Step {step} ---")
            print(f"Action: {action} | Phase: {phase_before} → {phase_after} {switched}")
            print(f"Obs: budget={obs['budget_remaining']:.3f}, fitness={obs['norm_fitness']:.4f}, "
                  f"stag={obs['stagnation']:.3f}, div={obs['diversity']:.3f}")
            print(f"Improvement: {'YES ✓' if improvement else 'NO'} | "
                  f"Raw fitness: {entry.get('raw_fitness_before', 'N/A'):.6f} → {entry.get('raw_fitness_after', 'N/A'):.6f}")
            print(f"Reward: {reward:+.4f} | Cumulative: {total_reward:+.4f}")
            print(f"Components: {', '.join(f'{k}={v:+.3f}' for k,v in components.items())}")
            
            if ended:
                print(f"\n{ended}")
        
        print(f"\n{'=' * 100}")
        print(f"TOTAL REWARD: {total_reward:+.4f}")
        print(f"{'=' * 100}")


def run_diagnostic(model_path: str = None, num_episodes: int = 3, 
                   max_decision_steps: int = 50, search_steps_per_decision: int = 10):
    """Run diagnostic episodes."""
    
    env = TracingOrchestratorEnv(
        max_decision_steps=max_decision_steps,
        search_steps_per_decision=search_steps_per_decision,
    )
    
    # Load trained model if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        use_model = True
    else:
        print("No model loaded - using random actions to compare")
        use_model = False
    
    for ep in range(num_episodes):
        print(f"\n{'#' * 100}")
        print(f"# EPISODE {ep + 1}")
        print(f"{'#' * 100}")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # For random baseline, use simple policy:
                # Stay until stagnation > 0.5, then advance
                stagnation = obs[2]
                action = 1 if stagnation > 0.5 else 0
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        
        env.print_trace()
        
        # Summary stats
        explore_steps = sum(1 for t in env.trace if 'phase_before' in t and t['phase_before'] == 'exploration')
        exploit_steps = sum(1 for t in env.trace if 'phase_before' in t and t['phase_before'] == 'exploitation')
        term_steps = sum(1 for t in env.trace if 'phase_before' in t and t['phase_before'] == 'termination')
        total_steps = len(env.trace) - 1
        
        print(f"\n📊 Episode Summary:")
        print(f"   Total Steps: {total_steps}")
        print(f"   Exploration: {explore_steps} ({100*explore_steps/max(1,total_steps):.1f}%)")
        print(f"   Exploitation: {exploit_steps} ({100*exploit_steps/max(1,total_steps):.1f}%)")
        print(f"   Termination: {term_steps}")


def compare_policies(max_decision_steps: int = 100, search_steps_per_decision: int = 20):
    """Compare different policies to understand optimal behavior."""
    
    print("\n" + "=" * 100)
    print("POLICY COMPARISON")
    print("=" * 100)
    
    results = {}
    
    # Policy 1: Always Stay (never advance)
    env = TracingOrchestratorEnv(
        max_decision_steps=max_decision_steps,
        search_steps_per_decision=search_steps_per_decision,
    )
    
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(0)  # Always STAY
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    results['always_stay'] = np.mean(rewards)
    
    # Policy 2: Always Advance
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(1)  # Always ADVANCE
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    results['always_advance'] = np.mean(rewards)
    
    # Policy 3: Stagnation-based switching
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            stagnation = obs[2]
            action = 1 if stagnation > 0.5 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    results['stagnation_based'] = np.mean(rewards)
    
    # Policy 4: Fixed 20-20 split
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        phase_idx = 0  # 0=explore, 1=exploit, 2=done (reset for each episode!)
        while not done:
            step += 1
            if phase_idx == 0 and step > 20:
                action = 1
                phase_idx = 1
            elif phase_idx == 1 and step > 40:
                action = 1
                phase_idx = 2
            else:
                action = 0
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    results['fixed_20_20'] = np.mean(rewards)
    
    print("\nPolicy Average Rewards:")
    for policy, avg_reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {policy:20s}: {avg_reward:+.3f}")
    
    print("\n⚠️ If 'always_advance' has high reward, the reward function incentivizes early exit!")
    print("⚠️ If 'stagnation_based' beats 'fixed_20_20', stagnation grows too fast!")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Episode Trace Diagnostic")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to trace")
    parser.add_argument("--max-decision-steps", type=int, default=50, help="Max decision steps")
    parser.add_argument("--search-steps", type=int, default=10, help="Search steps per decision")
    parser.add_argument("--compare", action="store_true", help="Compare different policies")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_policies(args.max_decision_steps, args.search_steps)
    else:
        run_diagnostic(
            model_path=args.model_path,
            num_episodes=args.episodes,
            max_decision_steps=args.max_decision_steps,
            search_steps_per_decision=args.search_steps,
        )
