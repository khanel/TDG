"""
Training script for RL policy using PPO on the orchestrator environment.
"""

import argparse
from stable_baselines3 import PPO
from ..environment import RLEnvironment
from ...core.orchestrator import Orchestrator
from ...solvers.registry import get_exploration_solver, get_exploitation_solver
from ...problems.registry import get_problem_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-solver", default="map_elites")
    parser.add_argument("--exploitation-solver", default="sa")
    parser.add_argument("--problem", default="tsp")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    args = parser.parse_args()

    # Instantiate problem and solvers via registries
    problem_class = get_problem_adapter(args.problem)
    exploration_class = get_exploration_solver(args.exploration_solver)
    exploitation_class = get_exploitation_solver(args.exploitation_solver)

    # Create instances (simplified, assume defaults)
    problem = problem_class()  # Assume default constructor
    exploration = exploration_class(problem)
    exploitation = exploitation_class(problem)
    orchestrator = Orchestrator(problem, exploration, exploitation)
    env = RLEnvironment(orchestrator)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.total_timesteps)
    model.save("ppo_orchestrator")


if __name__ == "__main__":
    main()