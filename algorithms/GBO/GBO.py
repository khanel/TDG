import numpy as np
from Core.search_algorithm import SearchAlgorithm
from Core.problem import Solution

class GradientBasedOptimizer(SearchAlgorithm):
    """
    Gradient-Based Optimizer (GBO) implementation.

    This algorithm uses two core operators:
    - GSR (Gradient Search Rule): guides solutions using population-informed search vectors
    - LEO (Local Escaping Operator): injects controlled randomness to escape local optima

    Based on Ahmadianfar et al. (2020) and subsequent variants.
    """

    def __init__(self, problem, population_size, alpha=1.0, beta=1.0, leo_prob=0.1,
                 max_iterations=100, step_size=1.0, verbosity=0,
                 exploration_boost=1.0, adaptive_exploration=False, **kwargs):
        """
        Initialize Gradient-Based Optimizer.

        Args:
            problem: The optimization problem
            population_size (int): Number of agents in population
            alpha (float): GSR coefficient for global best influence
            beta (float): GSR coefficient for population influence
            leo_prob (float): Probability of applying Local Escaping Operator
            max_iterations (int): Maximum number of iterations
            step_size (float): Step size for GSR movements
            verbosity (int): Verbosity level for logging
            exploration_boost (float): Multiplier for exploration enhancement
            adaptive_exploration (bool): Whether to adapt exploration over time
        """
        super().__init__(problem, population_size, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.leo_prob = leo_prob
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.verbosity = verbosity
        self.iteration = 0
        self.exploration_boost = exploration_boost
        self.adaptive_exploration = adaptive_exploration

        # GBO-specific attributes
        self.global_best = None
        self.global_best_fitness = float('inf')

        # Exploration tracking
        self.exploration_history = []
        self.best_fitness_history = []

    def initialize(self):
        """Initialize the population of agents."""
        super().initialize()

        # Get problem bounds
        problem_info = self.problem.get_problem_info()
        self.lower_bounds = np.array(problem_info.get('lower_bounds', [-np.inf]))
        self.upper_bounds = np.array(problem_info.get('upper_bounds', [np.inf]))

        # Find initial global best
        self._update_global_best()

    def _adapt_exploration_parameters(self):
        """Adapt exploration parameters based on iteration progress."""
        if not self.adaptive_exploration:
            return

        progress_ratio = self.iteration / self.max_iterations

        # Early exploration phase (first 30%): favor exploration
        if progress_ratio < 0.3:
            exploration_factor = self.exploration_boost
            self.alpha = 1.0 * exploration_factor  # Strong global influence
            self.beta = 1.0 * exploration_factor   # Strong population influence
            self.leo_prob = 0.2 * exploration_factor  # Higher escape probability
            self.step_size = 1.0 * exploration_factor

        # Mid phase (30-70%): balanced exploration/exploitation
        elif progress_ratio < 0.7:
            exploration_factor = 1.0 + (self.exploration_boost - 1.0) * (1 - progress_ratio)
            self.alpha = 1.0 * exploration_factor
            self.beta = 1.0 * exploration_factor
            self.leo_prob = 0.1 * exploration_factor
            self.step_size = 1.0

        # Late phase (70%+): favor exploitation
        else:
            self.alpha = 1.0
            self.beta = 1.0
            self.leo_prob = 0.05  # Lower escape probability
            self.step_size = 0.5   # Smaller steps for refinement

    def step(self):
        """Perform one iteration of GBO."""
        # Adapt exploration parameters
        self._adapt_exploration_parameters()

        new_population = []

        for i in range(self.population_size):
            current_agent = self.population[i]

            # Apply GSR (Gradient Search Rule)
            new_position = self._apply_gsr(current_agent, i)

            # Apply LEO (Local Escaping Operator)
            if np.random.random() < self.leo_prob:
                new_position = self._apply_leo(new_position, current_agent)

            # Ensure bounds are respected
            new_position = self._handle_bounds(new_position)

            # Create new solution and evaluate
            new_solution = Solution(new_position, self.problem)
            new_solution.evaluate()

            # Selection: keep the better solution
            if new_solution.fitness < current_agent.fitness:
                new_population.append(new_solution)
            else:
                new_population.append(current_agent)

        self.population = new_population
        self._update_global_best()
        self.iteration += 1

        # Track exploration metrics
        exploration_rate = self._calculate_exploration_rate()
        self.exploration_history.append(exploration_rate)
        self.best_fitness_history.append(self.global_best_fitness)

        # Verbosity logging
        if self.verbosity >= 1:
            print(f"Iteration {self.iteration}, Best Fitness: {self.global_best_fitness:.6f}, Exploration: {exploration_rate:.2f}")

    def _apply_gsr(self, agent, agent_idx):
        """
        Apply Gradient Search Rule (GSR).

        Builds a direction vector using population information to guide the search.
        """
        x = np.array(agent.representation)

        # Select random reference agents (different from current agent)
        available_indices = [j for j in range(self.population_size) if j != agent_idx]
        if len(available_indices) < 2:
            # If not enough agents, use global best as reference
            r1_pos = np.array(self.global_best)
            r2_pos = np.array(self.global_best)
        else:
            r1_idx, r2_idx = np.random.choice(available_indices, size=2, replace=False)
            r1_pos = np.array(self.population[r1_idx].representation)
            r2_pos = np.array(self.population[r2_idx].representation)

        # Global best position
        g_pos = np.array(self.global_best)

        # Build direction vector D using GSR formula
        # D = α * (g - x) + β * (r1 - r2)
        direction = (self.alpha * (g_pos - x) +
                    self.beta * (r1_pos - r2_pos))

        # Apply step size and move
        new_position = x + self.step_size * direction

        return new_position

    def _apply_leo(self, position, original_agent):
        """
        Apply Local Escaping Operator (LEO).

        Injects controlled randomness to escape local optima.
        """
        x = np.array(position)
        original_x = np.array(original_agent.representation)

        # Multiple escape strategies
        escape_type = np.random.choice(['gaussian', 'uniform', 'directional'], p=[0.4, 0.3, 0.3])

        if escape_type == 'gaussian':
            # Gaussian noise around current position
            noise_scale = 0.1 * np.mean(np.abs(self.upper_bounds - self.lower_bounds))
            noise = np.random.normal(0, noise_scale, size=len(x))
            escaped_position = x + noise

        elif escape_type == 'uniform':
            # Uniform perturbation within bounds
            if np.any(np.isfinite(self.lower_bounds)) and np.any(np.isfinite(self.upper_bounds)):
                perturbation = np.random.uniform(self.lower_bounds, self.upper_bounds)
                escaped_position = 0.7 * x + 0.3 * perturbation
            else:
                # For unbounded problems
                noise_scale = 0.5
                escaped_position = x + np.random.uniform(-noise_scale, noise_scale, size=len(x))

        else:  # directional
            # Directional escape toward global best or away from local region
            g_pos = np.array(self.global_best)

            # Mix of moving toward global best and random direction
            random_direction = np.random.normal(0, 0.5, size=len(x))
            escaped_position = x + 0.5 * (g_pos - x) + 0.3 * random_direction

        return escaped_position

    def _handle_bounds(self, position):
        """Handle boundary constraints."""
        if np.any(np.isfinite(self.lower_bounds)) and np.any(np.isfinite(self.upper_bounds)):
            # Clip to bounds
            return np.clip(position, self.lower_bounds, self.upper_bounds)
        else:
            # For unbounded problems, return as is
            return position

    def _update_global_best(self):
        """Update the global best solution."""
        for agent in self.population:
            if agent.fitness < self.global_best_fitness:
                self.global_best = agent.representation.copy()
                self.global_best_fitness = agent.fitness

        # Update the best_solution in parent class
        if self.global_best is not None:
            self.best_solution = Solution(self.global_best, self.problem)
            self.best_solution.fitness = self.global_best_fitness

    def _calculate_exploration_rate(self):
        """Calculate current exploration rate based on population diversity."""
        if len(self.population) < 2:
            return 0.0

        # Calculate average distance between agents and global best
        positions = np.array([agent.representation for agent in self.population])
        global_best_pos = np.array(self.global_best)

        # Diversity measure: average distance to global best
        distances_to_global = []
        for pos in positions:
            distances_to_global.append(np.linalg.norm(pos - global_best_pos))

        avg_distance = np.mean(distances_to_global)

        # Normalize by problem scale
        if np.any(np.isfinite(self.lower_bounds)) and np.any(np.isfinite(self.upper_bounds)):
            problem_scale = np.mean(self.upper_bounds - self.lower_bounds)
            exploration_rate = min(1.0, avg_distance / problem_scale)
        else:
            exploration_rate = min(1.0, avg_distance)

        return exploration_rate

    def get_gbo_info(self):
        """Get information about the current GBO state."""
        return {
            'iteration': self.iteration,
            'global_best_fitness': self.global_best_fitness,
            'global_best_position': self.global_best,
            'alpha': self.alpha,
            'beta': self.beta,
            'leo_prob': self.leo_prob,
            'step_size': self.step_size,
            'exploration_rate': self._calculate_exploration_rate() if self.exploration_history else 0.0
        }