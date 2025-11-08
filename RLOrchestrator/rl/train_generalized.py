"""
Generalized training script for the RL Orchestrator.
This script trains a single policy across multiple problems and solver combinations.
"""

import argparse
import random
import time
from typing import List, Type

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from RLOrchestrator.core.env_factory import create_env
from RLOrchestrator.problems.registry import get_problem_registry
from RLOrchestrator.solvers.registry import get_solver_registry, get_solvers
from Core.search_algorithm import SearchAlgorithm

def make_env_fn(problem_adapters: dict, solver_registry: dict, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        # Select a random problem
        problem_name = random.choice(list(problem_adapters.keys()))
        AdapterClass = problem_adapters[problem_name]
        
        # Select random solvers
        explorers = solver_registry.get(problem_name, {}).get('exploration', [])
        exploiters = solver_registry.get(problem_name, {}).get('exploitation', [])

        if not explorers or not exploiters:
            raise ValueError(f"No registered solvers for problem '{problem_name}'")

        ExplorerClass = random.choice(explorers)
        ExploiterClass = random.choice(exploiters)

        problem = AdapterClass()
        exploration_solver = ExplorerClass(problem)
        exploitation_solver = ExploiterClass(problem)
        
        env = create_env(
            problem=problem,
            exploration_solver=exploration_solver,
            exploitation_solver=exploitation_solver,
            session_id=seed,
            emit_init_summary=(rank==0)
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Generalized training for RL Orchestrator.")
    parser.add_argument("--total-timesteps", type=int, default=250000, help="Total timesteps for training.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--model-save-path", type=str, default="ppo_generalized.zip", help="Path to save the trained model.")
    args = parser.parse_args()

    problem_adapters = get_problem_registry()
    solver_registry = get_solver_registry()
    
    session_id = int(time.time())
    
    env_fns = [make_env_fn(problem_adapters, solver_registry, i, session_id) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save(args.model_save_path)

    print(f"Model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()