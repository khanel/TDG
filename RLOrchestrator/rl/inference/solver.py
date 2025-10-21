"""
Inference script to solve problems using trained RL policy.
"""

import argparse
from stable_baselines3 import PPO
from ..environment import RLEnvironment
from ...core.orchestrator import Orchestrator
from ...solvers.registry import get_exploration_solver, get_exploitation_solver
from ...problems.registry import get_problem_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ppo_orchestrator.zip")
    parser.add_argument("--exploration-solver", default="map_elites")
    parser.add_argument("--exploitation-solver", default="sa")
    parser.add_argument("--problem", default="tsp")
    args = parser.parse_args()

    # Load trained model and solve
    model = PPO.load(args.model_path)
    # Create environment as in training
    problem_class = get_problem_adapter(args.problem)
    exploration_class = get_exploration_solver(args.exploration_solver)
    exploitation_class = get_exploitation_solver(args.exploitation_solver)
    problem = problem_class()
    exploration = exploration_class(problem)
    exploitation = exploitation_class(problem)
    orchestrator = Orchestrator(problem, exploration, exploitation)
    env = RLEnvironment(orchestrator)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

    best = orchestrator.get_best_solution()
    print(f"Best solution: {best.representation}, Fitness: {best.fitness}")


if __name__ == "__main__":
    main()