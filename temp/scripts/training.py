
"""
Self-contained training script for the RL-based hyper-heuristic.

This script trains a PPO agent to orchestrate a 3-phase optimization pipeline:
- Phase 1 (Exploration): MAP-Elites for diverse solution discovery
- Phase 2 (Exploitation): Binary PSO for solution refinement
- Phase 3 (Termination): Episode ends, quality evaluated

Reward Function: Effectiveness-First Reward (EFR)
- EFFECTIVENESS FIRST: Quality gates everything (threshold = 0.7)
- EFFICIENCY SECOND: Budget savings only count if quality is good

Action semantics are UNIFIED across all phases:
- Action 0: STAY in current phase
- Action 1: ADVANCE to next phase
"""

import argparse
import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# All required components are in the local directory
from temp.core.base import OrchestratorEnv
from temp.core.reward import RewardWrapper, EFRConfig
from temp.nkl.problem import NKLAdapter
from temp.nkl.solvers import MAPElitesExplorer, BinaryPSO


class SelfContainedOrchestratorEnv(OrchestratorEnv):
    """
    An OrchestratorEnv with Effectiveness-First Reward (EFR).
    
    Key principle: Quality gates efficiency - no reward for fast garbage.
    """
    def __init__(self, n_items: int, k_interactions: int, population_size: int = 20, **kwargs):
        problem = NKLAdapter(n_items=n_items, k_interactions=k_interactions)
        
        # MAP-Elites for exploration (maintains diversity via behavioral archive)
        explorer = MAPElitesExplorer(problem, population_size=population_size, n_bins=10, mutation_rate=0.12)
        # Binary PSO for exploitation (converges toward global best)
        exploiter = BinaryPSO(problem, population_size=population_size, omega=0.7, c1=1.5, c2=2.0)

        super().__init__(
            problem=problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            **kwargs,
        )
        
        # Use Effectiveness-First Reward function
        # Quality threshold = 0.7, efficiency bonus only if quality is met
        self.reward_fn = RewardWrapper(EFRConfig())
        self._last_obs = None
        
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, action: int):
        # Store observation before step for reward calculation
        obs_before = self._last_obs
        
        # Base step runs the controller and gets the next observation
        obs, _, terminated, truncated, info = super().step(action)
        self._last_obs = obs.copy()
        
        # Calculate reward using Effectiveness-First Reward
        # Observation format: [budget_consumed, fitness_norm, improvement, stagnation, diversity, phase]
        signal = self.reward_fn.compute(obs_before, action, info)
        reward = signal.value
        
        return obs, reward, terminated, truncated, info


def make_env(args):
    """Utility function for vectorized environments."""
    def _init():
        return SelfContainedOrchestratorEnv(
            n_items=args.n_items,
            k_interactions=args.k_interactions,
            population_size=args.population_size,
            max_decision_steps=args.max_decision_steps,
            search_steps_per_decision=args.search_steps_per_decision,
            max_search_steps=args.max_search_steps,
        )
    return _init


def train(args):
    """Main training loop."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create a vectorized environment
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(args) for _ in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(args)])

    # Setup PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        device=args.device,
        n_steps=args.ppo_n_steps,
        batch_size=args.ppo_batch_size,
        n_epochs=args.ppo_epochs,
        learning_rate=args.ppo_lr,
        ent_coef=args.ppo_ent_coef,
        policy_kwargs={"net_arch": [64, 64]},
    )

    print("Starting training...")
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    
    # Save the final model
    os.makedirs("temp", exist_ok=True)
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Contained Trainer for RL-HH")
    parser.add_argument("--total-timesteps", type=int, default=25_000, help="Total timesteps for training.")
    parser.add_argument("--model-path", type=str, default="results/models/ppo_nkl_gwo_ga.zip", help="Path to save the trained model.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (e.g., 'cpu', 'cuda').")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--ppo-n-steps", type=int, default=2048, help="Number of steps to run for each environment per update.")
    parser.add_argument("--ppo-batch-size", type=int, default=64, help="Minibatch size for PPO.")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="Number of epochs when optimizing the surrogate loss.")
    parser.add_argument("--ppo-lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01, help="Entropy coefficient.")
    parser.add_argument("--max-decision-steps", type=int, default=200, help="Max agent decision steps per episode.")
    parser.add_argument("--search-steps-per-decision", type=int, default=10, help="Solver iterations per agent decision.")
    parser.add_argument("--max-search-steps", type=int, default=None, help="Optional hard cap on total solver iterations.")
    parser.add_argument("--n-items", type=int, default=100, help="NKL N: problem dimension.")
    parser.add_argument("--population-size", type=int, default=20, help="Population size for solvers.")
    parser.add_argument("--k-interactions", type=int, default=5, help="NKL K: number of epistatic interactions.")
    
    args = parser.parse_args()
    
    # Set the number of threads for PyTorch
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    
    train(args)
