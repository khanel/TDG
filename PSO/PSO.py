import numpy as np
from Core.search_algorithm import SearchAlgorithm
from Core.problem import Solution

class ParticleSwarmOptimization(SearchAlgorithm):
    """
    Particle Swarm Optimization (PSO) implementation.

    This algorithm simulates a swarm of particles that fly through the search space,
    influenced by their own best positions and the swarm's global best position.
    Supports both inertia weight and constriction factor formulations.
    """

    def __init__(self, problem, population_size, omega=0.7, c1=1.5, c2=1.5,
                 max_iterations=100, vmax_factor=0.2, use_constriction=False,
                 verbosity=0, exploration_boost=1.0, adaptive_exploration=False, **kwargs):
        """
        Initialize Particle Swarm Optimization.

        Args:
            problem: The optimization problem
            population_size (int): Number of particles in the swarm
            omega (float): Inertia weight (0.4-0.9, or None if using constriction)
            c1 (float): Cognitive coefficient (personal best influence)
            c2 (float): Social coefficient (global best influence)
            max_iterations (int): Maximum number of iterations
            vmax_factor (float): Maximum velocity factor (vmax = vmax_factor * (ub-lb))
            use_constriction (bool): Whether to use constriction factor formulation
            verbosity (int): Verbosity level for logging
            exploration_boost (float): Multiplier for exploration enhancement
            adaptive_exploration (bool): Whether to adapt exploration over time
        """
        super().__init__(problem, population_size, **kwargs)
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.vmax_factor = vmax_factor
        self.use_constriction = use_constriction
        self.verbosity = verbosity
        self.iteration = 0
        self.exploration_boost = exploration_boost
        self.adaptive_exploration = adaptive_exploration

        # PSO-specific attributes
        self.velocities = None
        self.personal_bests = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = float('inf')

        # Constriction factor calculation (if used)
        if self.use_constriction:
            phi = self.c1 + self.c2
            if phi <= 4:
                raise ValueError("For constriction PSO, c1 + c2 must be > 4")
            self.chi = 2.0 / (phi - 2 + np.sqrt(phi**2 - 4*phi))
            # Set omega to 1.0 for constriction (or None to indicate constriction mode)
            self.omega = 1.0

        # Exploration tracking
        self.exploration_history = []
        self.best_fitness_history = []

    def initialize(self):
        """Initialize the swarm of particles."""
        super().initialize()

        # Get problem bounds
        problem_info = self.problem.get_problem_info()
        self.lower_bounds = np.array(problem_info.get('lower_bounds', [-np.inf]))
        self.upper_bounds = np.array(problem_info.get('upper_bounds', [np.inf]))

        # Initialize velocities
        self.velocities = []
        for i in range(self.population_size):
            # Calculate vmax based on bounds
            if np.any(np.isfinite(self.lower_bounds)) and np.any(np.isfinite(self.upper_bounds)):
                vmax = self.vmax_factor * (self.upper_bounds - self.lower_bounds)
            else:
                # Default vmax for unbounded problems
                vmax = np.ones(len(self.population[0].representation)) * self.vmax_factor

            # Initialize velocity randomly within [-vmax, vmax]
            velocity = np.random.uniform(-vmax, vmax)
            self.velocities.append(velocity)

        # Initialize personal bests
        self.personal_bests = [sol.representation.copy() for sol in self.population]
        self.personal_best_fitness = [sol.fitness for sol in self.population]

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
            self.c1 = min(2.5, 1.5 * exploration_factor)  # Increase cognitive component
            self.c2 = min(2.5, 1.5 * exploration_factor * 0.8)  # Increase social component
            if not self.use_constriction:
                self.omega = min(0.9, 0.7 * exploration_factor)  # Higher inertia for exploration

        # Mid phase (30-70%): balanced exploration/exploitation
        elif progress_ratio < 0.7:
            exploration_factor = 1.0 + (self.exploration_boost - 1.0) * (1 - progress_ratio)
            self.c1 = 1.5 * exploration_factor
            self.c2 = 1.5 * exploration_factor
            if not self.use_constriction:
                self.omega = 0.7

        # Late phase (70%+): favor exploitation
        else:
            self.c1 = 1.5
            self.c2 = 1.5
            if not self.use_constriction:
                self.omega = 0.4  # Lower inertia for exploitation

    def step(self):
        """Perform one iteration of PSO."""
        # Adapt exploration parameters
        self._adapt_exploration_parameters()

        # Update each particle
        for i in range(self.population_size):
            # Get current position
            x = np.array(self.population[i].representation)

            # Update velocity
            if self.use_constriction:
                # Constriction factor formulation
                r1 = np.random.uniform(0, 1, size=len(x))
                r2 = np.random.uniform(0, 1, size=len(x))

                cognitive = self.c1 * r1 * (np.array(self.personal_bests[i]) - x)
                social = self.c2 * r2 * (np.array(self.global_best) - x)

                self.velocities[i] = self.chi * (np.array(self.velocities[i]) + cognitive + social)
            else:
                # Inertia weight formulation
                r1 = np.random.uniform(0, 1, size=len(x))
                r2 = np.random.uniform(0, 1, size=len(x))

                inertia = self.omega * np.array(self.velocities[i])
                cognitive = self.c1 * r1 * (np.array(self.personal_bests[i]) - x)
                social = self.c2 * r2 * (np.array(self.global_best) - x)

                self.velocities[i] = inertia + cognitive + social

            # Apply velocity clamping if bounds are defined
            if np.any(np.isfinite(self.lower_bounds)) and np.any(np.isfinite(self.upper_bounds)):
                vmax = self.vmax_factor * (self.upper_bounds - self.lower_bounds)
                self.velocities[i] = np.clip(self.velocities[i], -vmax, vmax)

            # Update position
            x_new = x + np.array(self.velocities[i])

            # Apply boundary handling
            x_new = self._handle_bounds(x_new)

            # Update particle position
            self.population[i] = Solution(x_new, self.problem)
            self.population[i].evaluate()

            # Update personal best
            if self.population[i].fitness < self.personal_best_fitness[i]:
                self.personal_bests[i] = x_new.copy()
                self.personal_best_fitness[i] = self.population[i].fitness

        # Update global best
        self._update_global_best()

        self.iteration += 1

        # Track exploration metrics
        exploration_rate = self._calculate_exploration_rate()
        self.exploration_history.append(exploration_rate)
        self.best_fitness_history.append(self.global_best_fitness)

        # Verbosity logging
        if self.verbosity >= 1:
            print(f"Iteration {self.iteration}, Best Fitness: {self.global_best_fitness:.6f}, Exploration: {exploration_rate:.2f}")

    def _handle_bounds(self, x):
        """Handle boundary constraints."""
        if np.any(np.isfinite(self.lower_bounds)) and np.any(np.isfinite(self.upper_bounds)):
            # Clip to bounds
            return np.clip(x, self.lower_bounds, self.upper_bounds)
        else:
            # For unbounded problems, return as is
            return x

    def _update_global_best(self):
        """Update the global best solution."""
        for i in range(self.population_size):
            if self.personal_best_fitness[i] < self.global_best_fitness:
                self.global_best = self.personal_bests[i].copy()
                self.global_best_fitness = self.personal_best_fitness[i]

        # Update the best_solution in parent class
        if self.global_best is not None:
            self.best_solution = Solution(self.global_best, self.problem)
            self.best_solution.fitness = self.global_best_fitness

    def _calculate_exploration_rate(self):
        """Calculate current exploration rate based on particle diversity."""
        if len(self.population) < 2:
            return 0.0

        # Calculate average distance between particles
        positions = np.array([sol.representation for sol in self.population])
        centroid = np.mean(positions, axis=0)

        distances = []
        for pos in positions:
            distances.append(np.linalg.norm(pos - centroid))

        avg_distance = np.mean(distances)

        # Normalize by problem scale
        if np.any(np.isfinite(self.lower_bounds)) and np.any(np.isfinite(self.upper_bounds)):
            problem_scale = np.mean(self.upper_bounds - self.lower_bounds)
            exploration_rate = min(1.0, avg_distance / problem_scale)
        else:
            exploration_rate = min(1.0, avg_distance)

        return exploration_rate

    def get_swarm_info(self):
        """Get information about the current swarm state."""
        return {
            'iteration': self.iteration,
            'global_best_fitness': self.global_best_fitness,
            'global_best_position': self.global_best,
            'avg_personal_best_fitness': np.mean(self.personal_best_fitness),
            'exploration_rate': self._calculate_exploration_rate() if self.exploration_history else 0.0,
            'omega': self.omega,
            'c1': self.c1,
            'c2': self.c2,
            'use_constriction': self.use_constriction,
            'chi': getattr(self, 'chi', None)
        }