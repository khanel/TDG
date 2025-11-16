import abc
from typing import List, Optional
from .problem import ProblemInterface, Solution # Use relative import

class SearchAlgorithm(abc.ABC):
    """
    Abstract base class for search algorithms.
    """
    # Optional metadata describing the solver's phase (exploration/exploitation).
    phase: Optional[str] = None

    def __init__(self, problem: ProblemInterface, population_size: int, **kwargs):
        """
        Initializes the search algorithm.

        Args:
            problem: An object implementing ProblemInterface.
            population_size: The size of the population to maintain.
            **kwargs: Algorithm-specific hyperparameters.
        """
        self.problem = problem
        self.population_size = population_size
        self.population: List[Solution] = []
        self.best_solution: Optional[Solution] = None
        self.iteration = 0
        # Store kwargs for algorithm-specific use
        self._config = kwargs

    def initialize(self):
        """
        Sets up the algorithm's initial state, including the population.
        Should be called before starting the search steps.
        """
        self.iteration = 0
        self.best_solution = None
        self.population = self.problem.get_initial_population(self.population_size)
        for sol in self.population:
            sol.evaluate() # Ensure initial fitness is calculated
        self._update_best_solution() # Find initial best

    @abc.abstractmethod
    def step(self):
        """
        Performs a single step (iteration/generation) of the search algorithm.
        This method should update the internal population and potentially the best_solution.
        """
        pass

    def _update_best_solution(self):
        """Updates the overall best solution found so far."""
        current_best_in_pop = min(self.population, default=None)
        if current_best_in_pop:
            if self.best_solution is None or current_best_in_pop < self.best_solution:
                # Create a copy to avoid modification if the population changes
                self.best_solution = Solution(current_best_in_pop.representation, self.problem)
                self.best_solution.fitness = current_best_in_pop.fitness


    def get_best_solution(self) -> Optional[Solution]:
        """
        Returns the best solution found by the algorithm so far.

        Returns:
            The best Solution object found, or None if the search hasn't started/found any.
        """
        return self.best_solution

    # --- Compatibility shims for observation interfaces ---
    def get_population(self) -> List[Solution]:
        """Return the current population (compat with RL observers)."""
        return self.population

    def get_best(self) -> Optional[Solution]:
        """Return best-so-far (compat name)."""
        return self.get_best_solution()

    def ingest_population(self, seeds: List[Solution]) -> None:
        """
        Default ingress for external seeds when switching stages.
        Copies the provided seeds and pads/repeats as necessary.
        """
        if not seeds:
            return

        clones: List[Solution] = []
        for sol in seeds:
            if sol is None:
                continue
            clone = sol.copy(preserve_id=False)
            # Ensure cloned solutions have evaluated fitness to avoid downstream laziness.
            if clone.fitness is None:
                clone.evaluate()
            clones.append(clone)

        if not clones:
            return

        base = list(clones)
        while len(clones) < self.population_size:
            template = base[len(clones) % len(base)]
            clones.append(template.copy(preserve_id=False))

        self.population = clones[: self.population_size]
        self._update_best_solution()

    def export_population(self) -> List[Solution]:
        """
        Return detached copies of the current population so consumers don't mutate our state.
        """
        exports: List[Solution] = []
        for sol in self.population:
            if sol is None:
                continue
            clone = sol.copy(preserve_id=False)
            exports.append(clone)
        return exports

    def ensure_population_evaluated(self) -> None:
        """Lazily evaluate any unevaluated individuals in-place."""
        for sol in self.population:
            if sol is not None and sol.fitness is None:
                sol.evaluate()

    # Optional: Add methods for convergence checks or state reporting
    # def is_converged(self, max_iterations: int, desired_fitness: float) -> bool: pass
    # def get_state(self) -> Dict[str, Any]: pass
