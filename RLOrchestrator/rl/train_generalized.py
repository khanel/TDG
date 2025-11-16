"""
Generalized training script for the RL Orchestrator.
This script trains a single policy across multiple problems and solver combinations.
"""

import argparse
import random
import time
from typing import List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from RLOrchestrator.core.env_factory import create_env
from RLOrchestrator.problems.registry import instantiate_problem, list_problem_definitions


def make_env_fn(problem_names: List[str], rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        problem_name = random.choice(problem_names)
        bundle = instantiate_problem(problem_name)
        stage_map = _stage_map(bundle.stages)
        for solver in stage_map.values():
            if hasattr(solver, "initialize"):
                solver.initialize()
        env = create_env(
            problem=bundle.problem,
            exploration_solver=stage_map["exploration"],
            exploitation_solver=stage_map["exploitation"],
            session_id=seed,
            log_type='train_generalized',
            emit_init_summary=(rank == 0),
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

    definitions = list_problem_definitions()
    problem_names = list(definitions.keys())
    
    session_id = int(time.time())
    
    env_fns = [make_env_fn(problem_names, i, session_id) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save(args.model_save_path)

    print(f"Model saved to {args.model_save_path}")


def _stage_map(stages):
    mapping = {binding.name: binding.solver for binding in stages}
    missing = {"exploration", "exploitation"} - mapping.keys()
    if missing:
        raise ValueError(f"Problem bundle missing stages: {sorted(missing)}")
    return mapping

if __name__ == "__main__":
    main()
