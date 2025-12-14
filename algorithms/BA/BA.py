import numpy as np
from Core.search_algorithm import SearchAlgorithm
from Core.problem import Solution

class BeesAlgorithm(SearchAlgorithm):
    """
    Bees Algorithm (BA) implementation.

    This algorithm mimics honeybee foraging behavior with:
    - Scout bees for exploration
    - Recruited bees for local search around promising sites
    - Elite sites get more recruited bees than non-elite sites
    """

    def __init__(self, problem, population_size, m, e, nep, nsp, ngh,
                 max_iterations=100, stlim=None, verbosity=0,
                 exploration_boost=1.5, adaptive_exploration=True, **kwargs):
        """
        Initialize Bees Algorithm with enhanced exploration.

        Args:
            problem: The optimization problem
            population_size (int): Total scout bees (n)
            m (int): Number of best sites selected for neighborhood search
            e (int): Elite sites among the selected sites
            nep (int): Recruited bees per elite site
            nsp (int): Recruited bees per non-elite selected site
            ngh (float): Neighborhood radius for local search
            max_iterations (int): Maximum number of iterations
            stlim (int, optional): Stagnation limit to abandon a site
            verbosity (int): Verbosity level for logging
            exploration_boost (float): Multiplier for exploration enhancement
            adaptive_exploration (bool): Whether to adapt exploration over time
        """
        super().__init__(problem, population_size, **kwargs)
        self.m = m
        self.e = e
        self.nep = nep
        self.nsp = nsp
        self.ngh = ngh
        self.max_iterations = max_iterations
        self.stlim = stlim
        self.verbosity = verbosity
        self.iteration = 0
        self.exploration_boost = exploration_boost
        self.adaptive_exploration = adaptive_exploration

        # Enhanced exploration parameters
        self.base_m = m
        self.base_e = e
        self.base_nep = nep
        self.base_nsp = nsp
        self.base_ngh = ngh

        # Track stagnation for each site (if stlim is used)
        if self.stlim is not None:
            self.stagnation_counters = [0] * population_size

        # Exploration tracking
        self.exploration_history = []
        self.best_fitness_history = []

    def initialize(self):
        """Initialize the population of scout bees."""
        super().initialize()
        if self.stlim is not None:
            self.stagnation_counters = [0] * self.population_size

    def _adapt_exploration_parameters(self):
        """Adapt exploration parameters based on iteration progress."""
        if not self.adaptive_exploration:
            return

        progress_ratio = self.iteration / self.max_iterations

        # Early exploration phase (first 30%): favor exploration
        if progress_ratio < 0.3:
            exploration_factor = self.exploration_boost
            self.m = max(2, int(self.base_m * exploration_factor))
            self.e = max(1, int(self.base_e * exploration_factor * 0.5))
            self.nep = max(1, int(self.base_nep * exploration_factor))
            self.nsp = max(1, int(self.base_nsp * exploration_factor))
            self.ngh = self.base_ngh * exploration_factor

        # Mid phase (30-70%): balanced exploration/exploitation
        elif progress_ratio < 0.7:
            exploration_factor = 1.0 + (self.exploration_boost - 1.0) * (1 - progress_ratio)
            self.m = max(2, int(self.base_m * exploration_factor))
            self.e = max(1, int(self.base_e * exploration_factor * 0.7))
            self.nep = max(1, int(self.base_nep * exploration_factor))
            self.nsp = max(1, int(self.base_nsp * exploration_factor))
            self.ngh = self.base_ngh * exploration_factor

        # Late phase (70%+): favor exploitation but maintain some exploration
        else:
            self.m = self.base_m
            self.e = self.base_e
            self.nep = self.base_nep
            self.nsp = self.base_nsp
            self.ngh = self.base_ngh

        # Ensure parameters don't exceed population size
        self.m = min(self.m, self.population_size - 1)
        self.e = min(self.e, self.m)

    def _apply_perturbation(self, population, perturbation_rate=0.1):
        """Apply random perturbation to maintain diversity."""
        if np.random.random() > perturbation_rate:
            return population

        # Perturb a subset of the population
        perturb_size = max(1, int(len(population) * 0.1))
        indices_to_perturb = np.random.choice(len(population), size=perturb_size, replace=False)

        for idx in indices_to_perturb:
            # Generate a completely new random solution
            population[idx] = self.problem.get_initial_solution()
            population[idx].evaluate()

        return population

    def step(self):
        """Perform one iteration of the Bees Algorithm with enhanced exploration."""
        # Adapt exploration parameters
        self._adapt_exploration_parameters()

        # Sort scouts by fitness (best first)
        self.population.sort(key=lambda x: x.fitness)

        # Apply perturbation to maintain diversity
        self.population = self._apply_perturbation(self.population)

        # Select top m sites (may be larger due to exploration boost)
        selected_sites = self.population[:self.m]

        # Mark top e as elite
        elite_sites = selected_sites[:self.e]
        non_elite_sites = selected_sites[self.e:]

        new_population = []

        # Process elite sites with enhanced local search
        for i, site in enumerate(elite_sites):
            best_site = self._enhanced_local_search(site, self.nep)
            new_population.append(best_site)

        # Process non-elite sites
        for i, site in enumerate(non_elite_sites):
            best_site = self._enhanced_local_search(site, self.nsp)
            new_population.append(best_site)

        # Enhanced global search for remaining bees
        remaining_bees = self.population_size - len(new_population)

        # Add some completely random solutions for exploration
        random_exploration_count = max(1, int(remaining_bees * 0.3))
        for _ in range(random_exploration_count):
            new_scout = self.problem.get_initial_solution()
            new_scout.evaluate()
            new_population.append(new_scout)

        # Add solutions based on best sites but with more perturbation
        for _ in range(remaining_bees - random_exploration_count):
            # Select a random site from the best half
            base_site = np.random.choice(self.population[:len(self.population)//2])
            # Generate a more distant neighbor
            new_scout = self._generate_distant_neighbor(base_site)
            new_scout.evaluate()
            new_population.append(new_scout)

        # Update stagnation counters if applicable
        if self.stlim is not None:
            self._update_stagnation_counters(new_population)

        self.population = new_population
        self._update_best_solution()
        self.iteration += 1

        # Track exploration metrics
        exploration_rate = (remaining_bees / self.population_size)
        self.exploration_history.append(exploration_rate)
        self.best_fitness_history.append(self.best_solution.fitness)

        # Verbosity logging
        if self.verbosity >= 1:
            print(f"Iteration {self.iteration}, Best Fitness: {self.best_solution.fitness}, Exploration: {exploration_rate:.2f}")

    def _local_search(self, site, num_neighbors):
        """
        Perform local search around a site by generating neighbors.

        Args:
            site: The site (Solution) to search around
            num_neighbors: Number of neighbors to generate

        Returns:
            The best solution found (site or its neighbors)
        """
        candidates = [site]  # Include the original site

        # Generate neighbors
        for _ in range(num_neighbors):
            neighbor = self._generate_neighbor(site)
            neighbor.evaluate()
            candidates.append(neighbor)

        # Return the best candidate
        return min(candidates, key=lambda x: x.fitness)

    def _enhanced_local_search(self, site, num_neighbors):
        """
        Enhanced local search with multiple neighborhood strategies.

        Args:
            site: The site (Solution) to search around
            num_neighbors: Number of neighbors to generate

        Returns:
            The best solution found
        """
        candidates = [site]  # Include the original site

        # Generate diverse neighbors using different strategies
        for i in range(num_neighbors):
            # Alternate between different neighbor generation strategies
            if i % 3 == 0:
                neighbor = self._generate_neighbor(site)  # Standard neighbor
            elif i % 3 == 1:
                neighbor = self._generate_distant_neighbor(site)  # More distant
            else:
                neighbor = self._generate_random_neighbor(site)  # Random direction

            neighbor.evaluate()
            candidates.append(neighbor)

        # Return the best candidate
        return min(candidates, key=lambda x: x.fitness)

    def _generate_distant_neighbor(self, site):
        """
        Generate a more distant neighbor for enhanced exploration.

        Args:
            site: The site (Solution) to generate neighbor around

        Returns:
            A more distant neighbor solution
        """
        problem_info = self.problem.get_problem_info()
        problem_type = problem_info.get('problem_type', 'continuous')

        if problem_type == 'discrete':
            return self._generate_distant_discrete_neighbor(site)
        else:
            return self._generate_distant_continuous_neighbor(site)

    def _generate_random_neighbor(self, site):
        """
        Generate a random neighbor in any direction.

        Args:
            site: The site (Solution) to generate neighbor around

        Returns:
            A random neighbor solution
        """
        problem_info = self.problem.get_problem_info()
        problem_type = problem_info.get('problem_type', 'continuous')

        if problem_type == 'discrete':
            # For discrete problems, generate a completely random solution
            return self.problem.get_initial_solution()
        else:
            return self._generate_distant_continuous_neighbor(site)

    def _generate_distant_continuous_neighbor(self, site):
        """Generate distant neighbor for continuous problems."""
        problem_info = self.problem.get_problem_info()
        lb = np.array(problem_info.get('lower_bounds', -np.inf))
        ub = np.array(problem_info.get('upper_bounds', np.inf))

        # Generate larger perturbation for distant neighbor
        perturbation = np.random.uniform(-self.ngh * 2, self.ngh * 2, size=len(site.representation))
        new_repr = np.array(site.representation) + perturbation

        # Ensure bounds are respected
        new_repr = np.clip(new_repr, lb, ub)

        return Solution(new_repr, self.problem)

    def _generate_distant_discrete_neighbor(self, site):
        """Generate distant neighbor for discrete problems using only safe operations."""
        new_repr = site.representation.copy()
        n = len(new_repr)

        # Perform multiple simple swap operations for distant neighbor
        num_operations = max(3, int(self.ngh * 2))

        for _ in range(num_operations):
            if n > 2:
                # Simple swap operation (excluding city 1 at index 0)
                i, j = np.random.choice(range(1, n), size=2, replace=False)
                new_repr[i], new_repr[j] = new_repr[j], new_repr[i]

        # Ensure city 1 is still at the start
        if new_repr[0] != 1:
            city1_idx = new_repr.index(1)
            new_repr[0], new_repr[city1_idx] = new_repr[city1_idx], new_repr[0]

        return Solution(new_repr, self.problem)

    def _generate_neighbor(self, site):
        """
        Generate a neighbor solution within ngh radius of the site.

        Args:
            site: The site (Solution) to generate neighbor around

        Returns:
            A new Solution object representing the neighbor
        """
        problem_info = self.problem.get_problem_info()
        problem_type = problem_info.get('problem_type', 'continuous')

        if problem_type == 'discrete':
            return self._generate_discrete_neighbor(site)
        else:
            return self._generate_continuous_neighbor(site)

    def _generate_continuous_neighbor(self, site):
        """Generate neighbor for continuous problems."""
        problem_info = self.problem.get_problem_info()
        lb = np.array(problem_info.get('lower_bounds', -np.inf))
        ub = np.array(problem_info.get('upper_bounds', np.inf))

        # Generate random perturbation within ngh radius
        perturbation = np.random.uniform(-self.ngh, self.ngh, size=len(site.representation))
        new_repr = np.array(site.representation) + perturbation

        # Ensure bounds are respected
        new_repr = np.clip(new_repr, lb, ub)

        return Solution(new_repr, self.problem)

    def _generate_discrete_neighbor(self, site):
        """Generate neighbor for discrete problems (e.g., TSP)."""
        # For discrete problems, implement a simple swap-based neighborhood
        new_repr = site.representation.copy()

        # Perform random swaps within ngh distance
        n = len(new_repr)
        if n > 2:  # Need at least 2 cities to swap
            # Choose ngh random swap operations
            for _ in range(min(int(self.ngh), n//2)):
                i, j = np.random.choice(range(1, n), size=2, replace=False)  # Skip index 0 if fixed start
                new_repr[i], new_repr[j] = new_repr[j], new_repr[i]

        return Solution(new_repr, self.problem)

    def _update_stagnation_counters(self, new_population):
        """Update stagnation counters for abandonment mechanism."""
        if self.stlim is None:
            return

        # Reset counters for improved sites, increment for unchanged
        for i, new_site in enumerate(new_population):
            if i < len(self.population):
                old_site = self.population[i]
                if new_site.fitness < old_site.fitness:
                    self.stagnation_counters[i] = 0
                else:
                    self.stagnation_counters[i] += 1

                    # Abandon site if stagnation limit reached
                    if self.stagnation_counters[i] >= self.stlim:
                        new_population[i] = self.problem.get_initial_solution()
                        new_population[i].evaluate()
                        self.stagnation_counters[i] = 0