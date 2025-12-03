#!/usr/bin/env python3
"""
Evaluation script for the trained RL-based hyper-heuristic agent on NKL.

This script evaluates the PPO model on the same domain it was trained on (NK-Landscape)
to measure in-domain performance before testing cross-domain generalization.
"""

import argparse
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from typing import List, Dict, Any

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from temp.core.base import OrchestratorEnv
from temp.nkl.problem import NKLAdapter
from temp.nkl.solvers import GrayWolfOptimization, GeneticAlgorithm

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationHarness:
    def __init__(self, args):
        self.args = args
        self.results_dir = "results"
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.reports_dir = os.path.join(self.results_dir, "reports")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def create_env(self, seed: int) -> OrchestratorEnv:
        """Creates a fresh environment instance."""
        problem = NKLAdapter(
            n_items=self.args.n_items,
            k_interactions=self.args.k_interactions,
            seed=seed
        )
        explorer = GrayWolfOptimization(problem, population_size=self.args.population_size)
        exploiter = GeneticAlgorithm(problem, population_size=self.args.population_size)
        
        return OrchestratorEnv(
            problem=problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=self.args.max_decision_steps,
            search_steps_per_decision=self.args.search_steps_per_decision,
            max_search_steps=self.args.max_search_steps
        )

    def run_agent(self, model: PPO, env: OrchestratorEnv, episode_num: int) -> Dict[str, Any]:
        """Runs the RL agent on the environment."""
        obs, _ = env.reset()
        done = False
        history = {'fitness': [], 'diversity': [], 'phase': [], 'action': [], 'stagnation': []}
        
        initial_fitness = env.get_best_solution().fitness
        
        steps = 0
        improvements = 0
        explore_steps = 0
        exploit_steps = 0
        switch_points = []  # exploration -> exploitation
        termination_step = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            current_fitness = env.get_best_solution().fitness
            current_phase = obs[4]  # 0.0=explore, 0.5=exploit, 1.0=terminate
            
            # Record metrics
            history['fitness'].append(current_fitness)
            history['diversity'].append(obs[3])
            history['stagnation'].append(obs[2])
            history['phase'].append(current_phase)
            history['action'].append(int(action))
            
            # Track phase switches
            if action == 1 and current_phase < 0.25:  # Advance from exploration
                switch_points.append(steps)
            elif action == 1 and 0.25 <= current_phase < 0.75:  # Advance from exploitation
                termination_step = steps
            
            # Count steps in each phase
            if current_phase < 0.25:
                explore_steps += 1
            elif current_phase < 0.75:
                exploit_steps += 1
            # termination phase doesn't count (episode ends)
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            new_fitness = env.get_best_solution().fitness
            if new_fitness < current_fitness:
                improvements += 1
        
        final_fitness = env.get_best_solution().fitness
        
        return {
            'history': history,
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'improvement': initial_fitness - final_fitness,
            'steps': steps,
            'improvements': improvements,
            'explore_steps': explore_steps,
            'exploit_steps': exploit_steps,
            'switch_points': switch_points,
            'termination_step': termination_step,
            'explore_ratio': explore_steps / max(1, explore_steps + exploit_steps)
        }

    def run_baseline(self, env: OrchestratorEnv, strategy: str) -> Dict[str, Any]:
        """Runs a baseline strategy."""
        obs, _ = env.reset()
        done = False
        history = {'fitness': [], 'diversity': [], 'phase': [], 'action': [], 'stagnation': []}
        
        initial_fitness = env.get_best_solution().fitness
        
        # Force initial phase for Pure Exploit
        if strategy == "pure_exploit":
            if obs[4] < 0.25:  # In exploration phase
                obs, _, _, _, _ = env.step(1)  # Advance to exploitation
        
        steps = 0
        improvements = 0
        
        while not done:
            current_phase = obs[4]  # 0.0=explore, 0.5=exploit, 1.0=terminate
            
            if strategy == "pure_explore":
                action = 0  # Never advance
            elif strategy == "pure_exploit":
                action = 0  # Stay in exploitation (already advanced)
            elif strategy == "random":
                action = np.random.choice([0, 1])
            elif strategy == "fixed_switch":
                # Advance at step 50 from exploration, at step 150 from exploitation
                if steps == 50 and current_phase < 0.25:
                    action = 1
                elif steps == 150 and 0.25 <= current_phase < 0.75:
                    action = 1
                else:
                    action = 0
            else:
                action = 0
            
            current_fitness = env.get_best_solution().fitness
            
            history['fitness'].append(current_fitness)
            history['diversity'].append(obs[3])
            history['stagnation'].append(obs[2])
            history['phase'].append(current_phase)
            history['action'].append(action)
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            new_fitness = env.get_best_solution().fitness
            if new_fitness < current_fitness:
                improvements += 1
        
        final_fitness = env.get_best_solution().fitness
        
        return {
            'history': history,
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'improvement': initial_fitness - final_fitness,
            'steps': steps,
            'improvements': improvements
        }

    def evaluate(self):
        """Main evaluation loop."""
        logger.info(f"Loading model from {self.args.model_path}")
        try:
            model = PPO.load(self.args.model_path, device=self.args.device)
        except FileNotFoundError:
            logger.error("Model file not found. Please train the agent first.")
            return

        metrics = {
            'agent': [],
            'pure_explore': [],
            'pure_exploit': [],
            'random': [],
            'fixed_switch': []
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION: NKL (N={self.args.n_items}, K={self.args.k_interactions})")
        logger.info(f"Episodes: {self.args.num_episodes}")
        logger.info(f"{'='*80}\n")
        
        for i in range(self.args.num_episodes):
            seed = self.args.seed + i
            
            # Run Agent
            env = self.create_env(seed)
            agent_result = self.run_agent(model, env, i)
            metrics['agent'].append(agent_result)
            
            # Run Baselines
            for strategy in ['pure_explore', 'pure_exploit', 'random', 'fixed_switch']:
                env = self.create_env(seed)
                baseline_result = self.run_baseline(env, strategy)
                metrics[strategy].append(baseline_result)
            
            # Progress logging
            if (i + 1) % 5 == 0 or i == 0:
                agent_fit = agent_result['final_fitness']
                explore_fit = metrics['pure_explore'][-1]['final_fitness']
                exploit_fit = metrics['pure_exploit'][-1]['final_fitness']
                logger.info(f"Episode {i+1:3d}/{self.args.num_episodes}: "
                           f"Agent={agent_fit:.4f}, Explore={explore_fit:.4f}, Exploit={exploit_fit:.4f}, "
                           f"Switch@{agent_result['switch_points']}")

        self.print_results(metrics)
        self.plot_results(metrics)

    def print_results(self, metrics: Dict[str, List[Dict]]):
        """Print comprehensive results table."""
        print("\n" + "=" * 100)
        print("EVALUATION RESULTS: NK-Landscape Problem")
        print("=" * 100)
        
        # Calculate stats
        stats = {}
        for strategy, runs in metrics.items():
            final_fits = [r['final_fitness'] for r in runs]
            improvements = [r['improvement'] for r in runs]
            stats[strategy] = {
                'mean_fit': np.mean(final_fits),
                'std_fit': np.std(final_fits),
                'best_fit': np.min(final_fits),
                'worst_fit': np.max(final_fits),
                'mean_imp': np.mean(improvements),
            }
        
        # Print table
        print(f"\n{'Strategy':<15} | {'Mean Fitness':>14} | {'Std':>8} | {'Best':>10} | {'Worst':>10} | {'Avg Improve':>12}")
        print("-" * 85)
        
        for strategy, s in stats.items():
            print(f"{strategy:<15} | {s['mean_fit']:>14.6f} | {s['std_fit']:>8.4f} | {s['best_fit']:>10.6f} | {s['worst_fit']:>10.6f} | {s['mean_imp']:>12.6f}")
        
        # Agent behavior analysis
        print("\n" + "-" * 85)
        print("AGENT BEHAVIOR ANALYSIS (3-Phase Pipeline)")
        print("-" * 85)
        
        agent_runs = metrics['agent']
        avg_explore_ratio = np.mean([r['explore_ratio'] for r in agent_runs])
        avg_switch_step = np.mean([r['switch_points'][0] if r['switch_points'] else r['steps'] for r in agent_runs])
        avg_termination_step = np.mean([r['termination_step'] if r['termination_step'] else r['steps'] for r in agent_runs])
        avg_explore_steps = np.mean([r['explore_steps'] for r in agent_runs])
        avg_exploit_steps = np.mean([r['exploit_steps'] for r in agent_runs])
        
        print(f"Average Exploration Steps: {avg_explore_steps:.1f}")
        print(f"Average Exploitation Steps: {avg_exploit_steps:.1f}")
        print(f"Average Exploration Ratio: {avg_explore_ratio:.1%}")
        print(f"Average Switch Step (Explore→Exploit): {avg_switch_step:.1f}")
        print(f"Average Termination Step (Exploit→Term): {avg_termination_step:.1f}")
        
        # Win rate
        agent_wins = 0
        total = len(metrics['agent'])
        for i in range(total):
            agent_fit = metrics['agent'][i]['final_fitness']
            best_baseline = min(
                metrics['pure_explore'][i]['final_fitness'],
                metrics['pure_exploit'][i]['final_fitness'],
                metrics['random'][i]['final_fitness'],
                metrics['fixed_switch'][i]['final_fitness']
            )
            if agent_fit <= best_baseline:
                agent_wins += 1
        
        print(f"\nAgent Win Rate vs All Baselines: {agent_wins}/{total} ({agent_wins/total:.1%})")
        
        # Comparison
        agent_mean = stats['agent']['mean_fit']
        best_baseline_mean = min(stats['pure_explore']['mean_fit'], 
                                  stats['pure_exploit']['mean_fit'],
                                  stats['random']['mean_fit'],
                                  stats['fixed_switch']['mean_fit'])
        
        if agent_mean < best_baseline_mean:
            improvement_pct = (best_baseline_mean - agent_mean) / abs(best_baseline_mean) * 100
            print(f"\n✅ AGENT WINS: {improvement_pct:.2f}% better than best baseline on average")
        else:
            gap_pct = (agent_mean - best_baseline_mean) / abs(best_baseline_mean) * 100
            print(f"\n⚠️  Agent is {gap_pct:.2f}% worse than best baseline")
        
        print("=" * 100)

    def plot_results(self, metrics: Dict[str, List[Dict]]):
        """Generate comparison plots."""
        # 1. Convergence Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Average Convergence
        ax1 = axes[0]
        colors = {'agent': 'blue', 'pure_explore': 'green', 'pure_exploit': 'red', 
                  'random': 'orange', 'fixed_switch': 'purple'}
        
        for strategy, runs in metrics.items():
            max_len = max(len(r['history']['fitness']) for r in runs)
            padded_fitness = []
            for r in runs:
                fit = r['history']['fitness']
                padded = fit + [fit[-1]] * (max_len - len(fit))
                padded_fitness.append(padded)
            
            mean_fitness = np.mean(padded_fitness, axis=0)
            std_fitness = np.std(padded_fitness, axis=0)
            
            steps = range(len(mean_fitness))
            ax1.plot(steps, mean_fitness, label=strategy, color=colors[strategy], linewidth=2)
            ax1.fill_between(steps, mean_fitness - std_fitness, mean_fitness + std_fitness, 
                           alpha=0.1, color=colors[strategy])
        
        ax1.set_title("Average Convergence Profile (NKL)")
        ax1.set_xlabel("Decision Steps")
        ax1.set_ylabel("Fitness (Lower is Better)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Best Agent Run Behavior
        ax2 = axes[1]
        best_run = min(metrics['agent'], key=lambda x: x['final_fitness'])
        history = best_run['history']
        
        steps = range(len(history['fitness']))
        ax2.plot(steps, history['fitness'], 'b-', label='Fitness', linewidth=2)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(steps, history['stagnation'], 'r--', label='Stagnation', alpha=0.7)
        ax2_twin.plot(steps, history['diversity'], 'g--', label='Diversity', alpha=0.7)
        ax2_twin.set_ylabel('Stagnation / Diversity', color='gray')
        
        # Phase background (3 phases: yellow=explore, cyan=exploit, magenta=terminate)
        phases = history['phase']
        for i in range(len(phases) - 1):
            if phases[i] < 0.25:
                color = 'yellow'  # Exploration
            elif phases[i] < 0.75:
                color = 'cyan'    # Exploitation  
            else:
                color = 'magenta' # Termination
            ax2.axvspan(i, i+1, facecolor=color, alpha=0.15)
        
        ax2.set_title(f"Best Agent Run (Final: {best_run['final_fitness']:.4f})")
        ax2.set_xlabel("Decision Steps")
        ax2.set_ylabel("Fitness", color='blue')
        
        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='yellow', alpha=0.3, label='Exploration'),
            Patch(facecolor='cyan', alpha=0.3, label='Exploitation'),
            Patch(facecolor='magenta', alpha=0.3, label='Termination'),
            Line2D([0], [0], color='b', label='Fitness'),
            Line2D([0], [0], color='r', linestyle='--', label='Stagnation'),
            Line2D([0], [0], color='g', linestyle='--', label='Diversity')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, "nkl_evaluation.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Plot saved to {save_path}")
        
        # 2. Box Plot Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = list(metrics.keys())
        data = [[r['final_fitness'] for r in metrics[s]] for s in strategies]
        
        bp = ax.boxplot(data, labels=strategies, patch_artist=True)
        
        for patch, strategy in zip(bp['boxes'], strategies):
            patch.set_facecolor(colors[strategy])
            patch.set_alpha(0.6)
        
        ax.set_title("Final Fitness Distribution by Strategy")
        ax.set_ylabel("Final Fitness (Lower is Better)")
        ax.grid(True, alpha=0.3, axis='y')
        
        save_path = os.path.join(self.plots_dir, "nkl_boxplot.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Box plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL-HH Agent on NKL")
    parser.add_argument("--model-path", type=str, default="results/models/ppo_nkl_gwo_ga.zip")
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--k-interactions", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-decision-steps", type=int, default=200)
    parser.add_argument("--search-steps-per-decision", type=int, default=10)
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    harness = EvaluationHarness(args)
    harness.evaluate()
