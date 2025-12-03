"""
Evaluation script for the trained RL-based hyper-heuristic agent on TSP.

This script loads the PPO model trained on NKL and evaluates its performance
on the Traveling Salesperson Problem (TSP) to test cross-domain generalization.
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

from temp.core.base import OrchestratorEnv, SearchAlgorithm, Solution, ProblemInterface
from temp.tsp.problem import TSPAdapter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- TSP Solvers ---

class TSPRandomSwapSearch(SearchAlgorithm):
    """
    Lightweight exploration solver for TSP that keeps permutations valid.
    It perturbs existing tours via random swaps to spread the search.
    """
    phase = "exploration"

    def __init__(self, problem: ProblemInterface, population_size: int, max_swaps: int = 3, **kwargs):
        super().__init__(problem, population_size, **kwargs)
        self.max_swaps = max_swaps

    def _perturb(self, tour: List[int]) -> List[int]:
        tour_copy = list(tour)
        num_swaps = np.random.randint(1, self.max_swaps + 1)
        for _ in range(num_swaps):
            i, j = np.random.choice(len(tour_copy), size=2, replace=False)
            tour_copy[i], tour_copy[j] = tour_copy[j], tour_copy[i]
        return tour_copy

    def step(self):
        self.ensure_population_evaluated()

        new_population = []
        # Keep the current best (elitism)
        if self.population:
            best = min(self.population, key=lambda s: s.fitness)
            new_population.append(best.copy(preserve_id=False))

        # Generate perturbed tours around existing individuals
        while len(new_population) < self.population_size:
            parent = np.random.choice(self.population)
            perturbed = self._perturb(parent.representation)
            new_population.append(Solution(perturbed, parent.problem))

        self.population = new_population[:self.population_size]
        self.ensure_population_evaluated()
        
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1


class TSPGeneticAlgorithm(SearchAlgorithm):
    """
    Exploitation-focused GA using order crossover and swap mutation for TSP permutations.
    """
    phase = "exploitation"

    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(problem, population_size, **kwargs)
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _tournament_select(self) -> Solution:
        candidates = np.random.choice(self.population, size=self.tournament_size, replace=False)
        best = min(candidates, key=lambda s: s.fitness)
        return best

    def _order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        size = len(p1)
        a, b = sorted(np.random.choice(range(size), size=2, replace=False))
        child = [-1] * size
        child[a:b] = p1[a:b]

        fill_pos = b % size
        for gene in p2:
            if gene in child:
                continue
            child[fill_pos] = gene
            fill_pos = (fill_pos + 1) % size
        return child

    def _swap_mutation(self, tour: List[int]) -> List[int]:
        i, j = np.random.choice(len(tour), size=2, replace=False)
        mutated = list(tour)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def step(self):
        self.ensure_population_evaluated()

        new_population = []
        # Elitism
        elite = min(self.population, key=lambda s: s.fitness)
        new_population.append(elite.copy(preserve_id=False))

        while len(new_population) < self.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if np.random.rand() < self.crossover_rate:
                child_repr = self._order_crossover(parent1.representation, parent2.representation)
            else:
                child_repr = list(parent1.representation)

            if np.random.rand() < self.mutation_rate:
                child_repr = self._swap_mutation(child_repr)

            new_population.append(Solution(child_repr, parent1.problem))

        self.population = new_population[:self.population_size]
        self.ensure_population_evaluated()
        
        self.mark_best_dirty()
        self._update_best_solution()
        self.iteration += 1


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
        problem = TSPAdapter(
            num_cities=self.args.num_cities,
            seed=seed
        )
        explorer = TSPRandomSwapSearch(problem, population_size=self.args.population_size)
        exploiter = TSPGeneticAlgorithm(problem, population_size=self.args.population_size)
        
        return OrchestratorEnv(
            problem=problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=self.args.max_decision_steps,
            search_steps_per_decision=self.args.search_steps_per_decision,
            max_search_steps=self.args.max_search_steps
        )

    def run_agent(self, model: PPO, env: OrchestratorEnv) -> Dict[str, Any]:
        """Runs the RL agent on the environment."""
        obs, _ = env.reset()
        done = False
        history = {'fitness': [], 'diversity': [], 'phase': [], 'action': []}
        
        initial_fitness = env.get_best_solution().fitness
        logger.info(f"Agent Run Started. Initial Fitness: {initial_fitness:.4f}")
        
        steps = 0
        improvements = 0
        explore_steps = 0
        exploit_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            current_fitness = env.get_best_solution().fitness
            
            # Record metrics
            history['fitness'].append(current_fitness)
            history['diversity'].append(obs[3])
            history['phase'].append(obs[4]) # 0.0 = Explore, 1.0 = Exploit
            history['action'].append(action)
            
            # Count steps in phase (based on observation before step)
            if obs[4] == 0.0:
                explore_steps += 1
            else:
                exploit_steps += 1
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            new_fitness = env.get_best_solution().fitness
            if new_fitness < current_fitness:
                improvements += 1
                # Log significant improvements or every improvement
                logger.info(f"  Step {steps}: Fitness improved {current_fitness:.4f} -> {new_fitness:.4f} (Phase: {'Exploit' if obs[4] == 1.0 else 'Explore'})")
            
        logger.info(f"Agent Run Finished. Steps: {steps}, Improvements: {improvements}")
        logger.info(f"Phase Split: Explore={explore_steps} ({explore_steps/steps*100:.1f}%), Exploit={exploit_steps} ({exploit_steps/steps*100:.1f}%)")
        logger.info(f"Final Fitness: {history['fitness'][-1]:.4f} (Total Improvement: {initial_fitness - history['fitness'][-1]:.4f})")
            
        return history

    def run_baseline(self, env: OrchestratorEnv, strategy: str) -> Dict[str, Any]:
        """Runs a baseline strategy (Pure Explore or Pure Exploit)."""
        obs, _ = env.reset()
        done = False
        history = {'fitness': [], 'diversity': [], 'phase': [], 'action': []}
        
        initial_fitness = env.get_best_solution().fitness
        logger.info(f"Baseline ({strategy}) Run Started. Initial Fitness: {initial_fitness:.4f}")
        
        # Force initial phase for Pure Exploit
        if strategy == "pure_exploit":
            if obs[4] == 0.0:
                obs, _, _, _, _ = env.step(1) # Switch
        
        steps = 0
        improvements = 0
        
        while not done:
            if strategy == "pure_explore":
                action = 0 # Stay in Exploration
            elif strategy == "pure_exploit":
                action = 0 # Stay in Exploitation
            
            current_fitness = env.get_best_solution().fitness
            
            history['fitness'].append(current_fitness)
            history['diversity'].append(obs[3])
            history['phase'].append(obs[4])
            history['action'].append(action)
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            new_fitness = env.get_best_solution().fitness
            if new_fitness < current_fitness:
                improvements += 1
                # logger.info(f"  Step {steps}: Fitness improved {current_fitness:.4f} -> {new_fitness:.4f}")
            
        logger.info(f"Baseline ({strategy}) Finished. Steps: {steps}, Improvements: {improvements}")
        logger.info(f"Final Fitness: {history['fitness'][-1]:.4f} (Total Improvement: {initial_fitness - history['fitness'][-1]:.4f})")
            
        return history

    def evaluate(self):
        """Main evaluation loop."""
        logger.info(f"Loading model from {self.args.model_path}")
        try:
            model = PPO.load(self.args.model_path, device=self.args.device)
        except FileNotFoundError:
            logger.error("Model file not found. Please train the agent first.")
            return

        metrics = {
            'agent': [], 'pure_explore': [], 'pure_exploit': []
        }
        
        best_agent_run_info = None
        best_agent_fitness = float('inf')
        
        logger.info(f"Starting evaluation on {self.args.num_episodes} episodes...")
        
        for i in range(self.args.num_episodes):
            seed = self.args.seed + i
            logger.info(f"Episode {i+1}/{self.args.num_episodes} (Seed: {seed})")
            
            # Run Agent
            env = self.create_env(seed)
            agent_hist = self.run_agent(model, env)
            metrics['agent'].append(agent_hist)
            
            # Track best run for plotting
            final_fitness = agent_hist['fitness'][-1]
            if final_fitness < best_agent_fitness:
                best_agent_fitness = final_fitness
                best_solution = env.get_best_solution()
                # Capture coordinates from the TSP problem instance
                city_coords = env.problem.tsp_problem.city_coords
                best_agent_run_info = {
                    'solution': best_solution,
                    'coords': city_coords,
                    'fitness': final_fitness,
                    'episode': i + 1
                }
            
            # Run Pure Explore
            env = self.create_env(seed)
            explore_hist = self.run_baseline(env, "pure_explore")
            metrics['pure_explore'].append(explore_hist)
            
            # Run Pure Exploit
            env = self.create_env(seed)
            exploit_hist = self.run_baseline(env, "pure_exploit")
            metrics['pure_exploit'].append(exploit_hist)

        self.generate_report(metrics)
        self.plot_results(metrics)
        
        if best_agent_run_info:
            self.plot_best_route(best_agent_run_info)

    def plot_best_route(self, run_info: Dict[str, Any]):
        """Plots the best route found by the agent."""
        solution = run_info['solution']
        coords = run_info['coords']
        fitness = run_info['fitness']
        episode = run_info['episode']
        
        tour = solution.representation # List of city indices
        
        # Reorder coords based on tour and close the loop
        tour_coords = [coords[i] for i in tour]
        tour_coords.append(tour_coords[0])
        
        xs, ys = zip(*tour_coords)
        
        plt.figure(figsize=(10, 10))
        plt.plot(xs, ys, 'b-', linewidth=1, alpha=0.7, label='Path')
        plt.plot(xs, ys, 'ro', markersize=5, label='Cities')
        
        # Mark start/end
        plt.plot(xs[0], ys[0], 'go', markersize=10, label='Start/End')
        
        plt.title(f"Best Agent Route (Episode {episode})\nLength: {fitness:.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, "best_agent_route.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Best route plot saved to {save_path}")

    def generate_report(self, metrics: Dict[str, List[Dict]]):
        """Generates a markdown report."""
        report_path = os.path.join(self.reports_dir, "evaluation_report.md")
        
        # Calculate aggregate stats
        stats = {}
        for strategy, runs in metrics.items():
            final_fitnesses = [run['fitness'][-1] for run in runs]
            stats[strategy] = {
                'mean': np.mean(final_fitnesses),
                'std': np.std(final_fitnesses),
                'min': np.min(final_fitnesses),
                'max': np.max(final_fitnesses)
            }
            
        with open(report_path, "w") as f:
            f.write("# RL-Orchestrator Evaluation Report (TSP)\n\n")
            f.write(f"**Problem:** TSP (Cities={self.args.num_cities})\n")
            f.write(f"**Episodes:** {self.args.num_episodes}\n\n")
            
            f.write("## Performance Summary (Final Tour Length)\n")
            f.write("| Strategy | Mean | Std Dev | Best (Min) | Worst (Max) |\n")
            f.write("|----------|------|---------|------------|-------------|\n")
            for strat, s in stats.items():
                f.write(f"| **{strat}** | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |\n")
            
            f.write("\n## Analysis\n")
            agent_mean = stats['agent']['mean']
            explore_mean = stats['pure_explore']['mean']
            exploit_mean = stats['pure_exploit']['mean']
            
            if agent_mean < explore_mean and agent_mean < exploit_mean:
                f.write("✅ **Success:** The RL Agent outperformed both baselines on average.\n")
            elif agent_mean < explore_mean:
                f.write("⚠️ **Partial Success:** The RL Agent beat Pure Exploration but not Pure Exploitation.\n")
            else:
                f.write("❌ **Failure:** The RL Agent failed to beat the baselines.\n")

        logger.info(f"Report saved to {report_path}")

    def plot_results(self, metrics: Dict[str, List[Dict]]):
        """Generates comparison plots."""
        # 1. Convergence Plot (Average over episodes)
        plt.figure(figsize=(10, 6))
        
        for strategy, runs in metrics.items():
            # Pad histories to same length for averaging
            max_len = max(len(r['fitness']) for r in runs)
            padded_fitness = []
            for r in runs:
                fit = r['fitness']
                padded = fit + [fit[-1]] * (max_len - len(fit))
                padded_fitness.append(padded)
            
            mean_fitness = np.mean(padded_fitness, axis=0)
            std_fitness = np.std(padded_fitness, axis=0)
            
            steps = range(len(mean_fitness))
            plt.plot(steps, mean_fitness, label=strategy, linewidth=2)
            plt.fill_between(steps, mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.1)
            
        plt.title("Average Convergence Profile (TSP)")
        plt.xlabel("Decision Steps")
        plt.ylabel("Tour Length (Lower is Better)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, "convergence_comparison.png"))
        plt.close()
        
        # 2. Agent Behavior (Phase Distribution)
        # Take the best agent run
        best_agent_run = min(metrics['agent'], key=lambda x: x['fitness'][-1])
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        steps = range(len(best_agent_run['fitness']))
        ax1.plot(steps, best_agent_run['fitness'], 'b-', label='Fitness')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Fitness', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(steps, best_agent_run['diversity'], 'g--', label='Diversity', alpha=0.6)
        ax2.set_ylabel('Diversity', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Background for phases
        phases = best_agent_run['phase']
        # 0 = Explore, 1 = Exploit
        # Create colored spans
        for i in range(len(phases) - 1):
            color = 'yellow' if phases[i] == 0.0 else 'orange'
            alpha = 0.2
            ax1.axvspan(i, i+1, facecolor=color, alpha=alpha)
            
        plt.title("Best Agent Run: Fitness, Diversity & Phases")
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='yellow', alpha=0.2, label='Exploration'),
            Patch(facecolor='orange', alpha=0.2, label='Exploitation'),
            plt.Line2D([0], [0], color='b', label='Fitness'),
            plt.Line2D([0], [0], color='g', linestyle='--', label='Diversity')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(os.path.join(self.plots_dir, "agent_behavior.png"))
        plt.close()
        
        logger.info(f"Plots saved to {self.plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL-HH Agent on TSP")
    parser.add_argument("--model-path", type=str, default="results/models/ppo_nkl_gwo_ga.zip")
    parser.add_argument("--num-cities", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-decision-steps", type=int, default=200)
    parser.add_argument("--search-steps-per-decision", type=int, default=10)
    parser.add_argument("--max-search-steps", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    harness = EvaluationHarness(args)
    harness.evaluate()
