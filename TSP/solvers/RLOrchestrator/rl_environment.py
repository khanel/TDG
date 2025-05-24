import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Try relative imports first, fallback to absolute imports, then fallback to mock strategies
try:
    from .exploration_strategies import create_exploration_strategy
    from .exploitation_strategies import create_exploitation_strategy
except ImportError:
    try:
        from exploration_strategies import create_exploration_strategy
        from exploitation_strategies import create_exploitation_strategy
    except ImportError:
        # Fallback to mock strategies for testing
        try:
            from .mock_strategies import create_exploration_strategy, create_exploitation_strategy
        except ImportError:
            from mock_strategies import create_exploration_strategy, create_exploitation_strategy

class RLEnv(gym.Env):
    """
    Custom Environment for RL-based Orchestration of Search Algorithms.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    # Reward component constants (can be tuned)
    REWARD_IMPROVEMENT_FACTOR = 10.0
    REWARD_NO_IMPROVEMENT_PENALTY_STEP = -2.0  # Flat penalty for a step that yields no improvement
    REWARD_ITERATION_COST_FACTOR = -0.05  # Cost per iteration spent in the action

    TERMINATION_REWARD_BASE_FITNESS_FACTOR = 200.0  # e.g., 200 / (1 + fitness)
    TERMINATION_REWARD_BUDGET_USAGE_PENALTY_FACTOR = 20.0  # Penalty for (iter_used / total_iter) * factor

    def __init__(self, problem_instance, total_max_iterations, iteration_options=None, 
                 exploration_strategy_type="ga", exploitation_strategy_type="local_search"):
        super(RLEnv, self).__init__()

        self.problem_instance = problem_instance
        self.total_max_iterations = total_max_iterations
        self.iteration_options = iteration_options if iteration_options is not None else [10, 20, 50, 100]
        
        # Current state of the environment
        self.current_iterations_passed = 0
        self.best_fitness_so_far = float('inf') # Assuming minimization problem
        self.best_solution_so_far = None  # Track best solution for exploitation

        # Define action space:
        # Action is a tuple: (strategy_choice, iteration_budget_choice_index)
        # strategy_choice: 0=Explore, 1=Exploit, 2=Terminate
        # iteration_budget_choice_index: index into self.iteration_options
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),  # Strategy: Explore, Exploit, Terminate
            spaces.Discrete(len(self.iteration_options))  # Index for iteration budget
        ))

        # Define observation space:
        # [current_iterations_passed, best_fitness_so_far, iterations_remaining]
        # We use float32 for compatibility with SB3
        # Max values can be set based on problem specifics if known, else use np.inf
        # For iterations_remaining, it's total_max_iterations - current_iterations_passed
        # For best_fitness_so_far, it can be unbounded or bounded if problem specifics allow
        
        # Let's define the observation space components:
        # 1. Normalized current_iterations_passed (0 to 1)
        # 2. Normalized best_fitness_so_far (e.g., 0 to 1, requires min/max estimates or dynamic scaling)
        # 3. Normalized iterations_remaining (0 to 1)
        # For simplicity now, let's use raw values and consider normalization later.
        
        # Observation: [current_iterations_passed, best_fitness_so_far]
        # We'll derive iterations_remaining or pass total_max_iterations separately if needed by the agent,
        # or include it in the observation.
        # As per plan: Total Iterations, Current Iterations Passed, Best Solution Fitness.
        # Let's assume Total Iterations is fixed for an episode and known.
        # So, observation: [Current Iterations Passed, Best Solution Fitness]
        # For now, let's use a simple observation space.
        # Low and High bounds for observations:
        # current_iterations_passed: [0, total_max_iterations]
        # best_fitness_so_far: [-inf, inf] (or more specific if problem allows)
        obs_low = np.array([0, -np.inf], dtype=np.float32)
        obs_high = np.array([self.total_max_iterations, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize exploration and exploitation strategy handlers
        self.exploration_strategy = create_exploration_strategy(exploration_strategy_type)
        self.exploitation_strategy = create_exploitation_strategy(exploitation_strategy_type)

    def _get_obs(self):
        """Helper function to get the current observation."""
        return np.array([self.current_iterations_passed, self.best_fitness_so_far], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_iterations_passed = 0
        self.best_fitness_so_far = float('inf')
        self.best_solution_so_far = None
        # TODO: Reset problem state if necessary (e.g., clear solution caches in underlying algos)
        # TODO: Reset exploration_strategy and exploitation_strategy if they have internal state
        
        observation = self._get_obs()
        info = {} # Additional info (optional)
        return observation, info

    def step(self, action):
        terminated = False
        truncated = False # For time limits not part of the MDP
        reward = 0
        info = {} # Additional info (optional)

        # Check for termination due to budget exhaustion *before* processing the current action
        if self.current_iterations_passed >= self.total_max_iterations:
            terminated = True
            reward = self._calculate_termination_reward(self.best_fitness_so_far, self.current_iterations_passed)
            observation = self._get_obs()
            info['status'] = "terminated_max_iterations_reached_before_action"
            return observation, reward, terminated, truncated, info

        strategy_choice, iteration_budget_idx = action
        action_iteration_budget = 0
        if strategy_choice < 2: # Explore or Exploit
            action_iteration_budget = self.iteration_options[iteration_budget_idx]

        new_fitness_from_strategy = self.best_fitness_so_far
        actual_iterations_spent_by_strategy = 0

        if strategy_choice == 0: # Explore
            # Execute exploration strategy
            try:
                actual_iterations_spent_by_strategy, new_fitness_from_strategy = self.exploration_strategy.run(
                    self.problem_instance, action_iteration_budget, self.best_fitness_so_far
                )
                # Update best solution if exploration found a better one
                # Note: We don't get the actual solution from the current interface
                # This could be enhanced in the future to also return the solution
            except Exception as e:
                # Fallback to mock behavior if strategy fails
                print(f"Exploration strategy failed: {e}, using mock behavior")
                actual_iterations_spent_by_strategy = action_iteration_budget
                if np.random.rand() < 0.6:  # 60% chance of some change
                    change = (np.random.rand() - 0.4) * 20
                    new_fitness_from_strategy = max(0, self.best_fitness_so_far - change)
                else:
                    new_fitness_from_strategy = self.best_fitness_so_far
            
            reward = self._calculate_reward(self.best_fitness_so_far, new_fitness_from_strategy, actual_iterations_spent_by_strategy)
            if new_fitness_from_strategy < self.best_fitness_so_far:
                self.best_fitness_so_far = new_fitness_from_strategy
                # TODO: Also update best_solution_so_far when we get solutions from strategies
            self.current_iterations_passed += actual_iterations_spent_by_strategy
            info['status'] = "explore_step"

        elif strategy_choice == 1: # Exploit
            # Execute exploitation strategy
            try:
                actual_iterations_spent_by_strategy, new_fitness_from_strategy = self.exploitation_strategy.run(
                    self.problem_instance, action_iteration_budget, self.best_fitness_so_far, self.best_solution_so_far
                )
            except Exception as e:
                # Fallback to mock behavior if strategy fails
                print(f"Exploitation strategy failed: {e}, using mock behavior")
                actual_iterations_spent_by_strategy = action_iteration_budget # Assume it uses the full budget
                if np.random.rand() < 0.8: # 80% chance of some change (exploitation is more directed)
                    change = (np.random.rand() - 0.2) * 10 # Simulates smaller, more certain improvement
                    new_fitness_from_strategy = max(0, self.best_fitness_so_far - change)
                else: # 20% chance no change
                    new_fitness_from_strategy = self.best_fitness_so_far

            reward = self._calculate_reward(self.best_fitness_so_far, new_fitness_from_strategy, actual_iterations_spent_by_strategy)
            if new_fitness_from_strategy < self.best_fitness_so_far:
                self.best_fitness_so_far = new_fitness_from_strategy
            self.current_iterations_passed += actual_iterations_spent_by_strategy
            info['status'] = "exploit_step"
            
        elif strategy_choice == 2: # Terminate by agent's choice
            terminated = True
            # Iterations spent for this "terminate" action is 0.
            # The reward is based on the state *before* this termination action.
            reward = self._calculate_termination_reward(self.best_fitness_so_far, self.current_iterations_passed)
            info['status'] = "terminate_action_chosen"
            # No change in iterations_passed or best_fitness from this action itself.

        # Ensure current_iterations_passed does not exceed total_max_iterations (clamp)
        # and check for termination if budget was exhausted by the explore/exploit action
        if self.current_iterations_passed >= self.total_max_iterations:
            terminated = True # Mark as terminated if budget exhausted
            self.current_iterations_passed = self.total_max_iterations # Clamp
            if info.get('status') not in ["terminate_action_chosen", "terminated_max_iterations_reached_before_action"]:
                 # If terminated due to budget overrun by explore/exploit, the reward for that action stands.
                 # We might add a specific note to info.
                info['status'] = info.get('status', '') + "_max_iterations_reached_after_action"


        observation = self._get_obs()
        # info already populated
        
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, old_fitness, new_fitness, iterations_spent_for_action):
        improvement = old_fitness - new_fitness  # Assuming minimization
        
        # Cost for iterations spent
        iteration_cost_penalty = iterations_spent_for_action * self.REWARD_ITERATION_COST_FACTOR

        if improvement > 0:
            # Reward for improvement, scaled
            improvement_reward = improvement * self.REWARD_IMPROVEMENT_FACTOR
            # Optional: Add efficiency bonus here if desired
            # efficiency_bonus = (improvement / iterations_spent_for_action) * self.SOME_EFFICIENCY_FACTOR if iterations_spent_for_action > 0 else 0
            return improvement_reward + iteration_cost_penalty
        else:
            # Penalty for no improvement or worsening step
            return self.REWARD_NO_IMPROVEMENT_PENALTY_STEP + iteration_cost_penalty

    def _calculate_termination_reward(self, final_fitness, total_iterations_used_episode):
        # Reward based on final fitness (higher is better for 1/(1+fitness) type scaling)
        # Add 1 to fitness to prevent division by zero if fitness can be 0 and handle negative fitness if possible.
        # Assuming fitness is non-negative.
        fitness_component = self.TERMINATION_REWARD_BASE_FITNESS_FACTOR / (1 + max(0, final_fitness))

        # Penalty based on normalized budget usage
        budget_usage_ratio = total_iterations_used_episode / self.total_max_iterations
        budget_penalty_component = budget_usage_ratio * self.TERMINATION_REWARD_BUDGET_USAGE_PENALTY_FACTOR
        
        return fitness_component - budget_penalty_component

    def render(self):
        # TODO: Implement visualization if needed (e.g., plot fitness over iterations)
        if self.metadata['render_modes'].__contains__('human'):
            print(f"Iterations: {self.current_iterations_passed}, Best Fitness: {self.best_fitness_so_far}")
        else:
            pass # No rendering

    def close(self):
        # TODO: Clean up any resources if needed
        pass

# Example Usage (for testing the environment structure)
if __name__ == '__main__':
    # This is a placeholder for problem_instance.
    # In a real scenario, you'd load or define your TSP problem here.
    class MockProblem:
        def __init__(self):
            self.name = "mock_tsp"
            # Add any other attributes your environment might need from the problem

    mock_problem = MockProblem()
    iteration_budgets = [10, 50, 100] # Example iteration options
    env = RLEnv(problem_instance=mock_problem, total_max_iterations=1000, iteration_options=iteration_budgets)

    print("Action Space:", env.action_space)
    print("Action Space Sample:", env.action_space.sample())
    print("Observation Space:", env.observation_space)
    print("Observation Space Sample:", env.observation_space.sample())

    obs, info = env.reset()
    print("Initial Observation:", obs)

    for _ in range(10): # Increased range for more steps
        action = env.action_space.sample() # Sample a random action
        print(f"Sampled Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print("Episode finished.")
            obs, info = env.reset()
            print("Initial Observation after reset:", obs)
    env.close()
