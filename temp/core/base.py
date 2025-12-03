
"""
This file is a self-contained module created by refactoring various
components from the original project into a single place for simplicity.
It contains the core logic for the reinforcement learning environment,
including the problem definition, search algorithms, and orchestration.
"""

import abc
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Dict, List, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np


# --- Placeholder for logging utility ---

def setup_logging(log_type: str, problem_label: str, log_dir: str, session_id: Optional[int]) -> logging.Logger:
    """A placeholder logging setup."""
    logger = logging.getLogger(f"{log_type}_{problem_label}")
    # Avoid adding handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --- From Core/problem.py ---

class ProblemInterface(abc.ABC):
    """
    Abstract base class defining the interface for an optimization problem.
    """

    @abc.abstractmethod
    def evaluate(self, solution: 'Solution') -> float:
        """
        Evaluates the fitness of a given solution. Lower values are better.
        """
        pass

    @abc.abstractmethod
    def get_initial_solution(self) -> 'Solution':
        """
        Generates a single, potentially random, valid initial solution.
        """
        pass

    @abc.abstractmethod
    def get_problem_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing essential information about the problem.
        """
        pass

    def get_bounds(self) -> Dict[str, Any]:
        """
        Optional hook to expose domain bounds.
        """
        return {}

    def regenerate_instance(self) -> bool:
        """
        Optional hook to randomize the problem instance.
        """
        return False

    def get_initial_population(self, population_size: int) -> List['Solution']:
        """
        Generates an initial population of solutions using vectorized evaluation.
        """
        population = [self.get_initial_solution() for _ in range(population_size)]
        
        # Use vectorized batch evaluation if available
        if hasattr(self, 'batch_evaluate_solutions'):
            self.batch_evaluate_solutions(population)
        else:
            # Fallback to individual evaluation
            for sol in population:
                sol.evaluate()
        
        return population


class Solution:
    """Represents a potential solution to the optimization problem."""
    _id_counter = count()

    def __init__(self, representation: Any, problem: ProblemInterface, *, solution_id: Optional[int] = None):
        self.representation = representation
        self.problem = problem
        self.fitness: Optional[float] = None
        self.id: int = int(next(self._id_counter) if solution_id is None else solution_id)

    def evaluate(self):
        """Calculates and stores the fitness of this solution."""
        if self.fitness is None:
            self.fitness = self.problem.evaluate(self)
        return self.fitness

    def copy(self, *, preserve_id: bool = True) -> 'Solution':
        """Creates a safe copy of this solution without expensive deepcopy."""
        new_id = self.id if preserve_id else None
        
        # Handle different representation types safely
        if hasattr(self.representation, 'copy'):
            # NumPy arrays - use .copy() for fast, safe duplication
            new_repr = self.representation.copy()
        elif isinstance(self.representation, list):
            # Python lists - use slice copy for safe duplication
            new_repr = self.representation.copy()
        else:
            # Fallback to deepcopy for unknown types
            import copy
            new_repr = copy.deepcopy(self.representation)
        
        new_solution = Solution(new_repr, self.problem, solution_id=new_id)
        new_solution.fitness = self.fitness
        return new_solution

    def __lt__(self, other: 'Solution') -> bool:
        """Allows comparison based on fitness (assuming minimization)."""
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness < other.fitness


# --- From Core/search_algorithm.py ---

class SearchAlgorithm(abc.ABC):
    """
    Abstract base class for search algorithms.
    """
    phase: Optional[str] = None

    def __init__(self, problem: ProblemInterface, population_size: int, **kwargs):
        self.problem = problem
        self.population_size = population_size
        self.population: List[Solution] = []
        self.best_solution: Optional[Solution] = None
        self.iteration = 0
        self._config = kwargs
        self._cached_best = None
        self._best_dirty = True

    def initialize(self):
        """Sets up the algorithm's initial state."""
        self.iteration = 0
        self.best_solution = None
        self.population = self.problem.get_initial_population(self.population_size)
        for sol in self.population:
            sol.evaluate()
        self._update_best_solution()

    @abc.abstractmethod
    def step(self):
        """Performs a single step of the search algorithm."""
        pass

    def _update_best_solution(self):
        """Updates the overall best solution found so far with caching."""
        if self._best_dirty or self._cached_best is None:
            current_best_in_pop = min(self.population, default=None)
            if current_best_in_pop:
                if self.best_solution is None or current_best_in_pop < self.best_solution:
                    self.best_solution = current_best_in_pop.copy(preserve_id=True)
                self._cached_best = current_best_in_pop
            self._best_dirty = False
    
    def mark_best_dirty(self):
        """Mark the cached best solution as dirty (needs recalculation)."""
        self._best_dirty = True

    def get_best(self) -> Optional[Solution]:
        """Returns the best solution found so far."""
        return self.best_solution

    def get_population(self) -> List[Solution]:
        """Returns the current population."""
        return self.population

    def ingest_population(self, seeds: List[Solution]):
        """Ingests a population from a previous stage."""
        if not seeds:
            self.initialize()
            return
        
        clones = [sol.copy(preserve_id=False) for sol in seeds if sol]
        
        while len(clones) < self.population_size:
            clones.extend(clones) # Simple extension
        
        self.population = clones[:self.population_size]
        self.mark_best_dirty()  # Mark dirty since population changed
        self.ensure_population_evaluated()
        self._update_best_solution()

    def export_population(self) -> List[Solution]:
        """Exports the current population."""
        return [sol.copy(preserve_id=False) for sol in self.population]

    def ensure_population_evaluated(self):
        """Ensures all individuals in the population have a fitness value using vectorized evaluation."""
        # Find solutions that need evaluation
        unevaluated = [sol for sol in self.population if sol.fitness is None]
        
        if unevaluated:
            # Use vectorized batch evaluation if available
            if hasattr(self.problem, 'batch_evaluate_solutions'):
                self.problem.batch_evaluate_solutions(unevaluated)
            else:
                # Fallback to individual evaluation
                for sol in unevaluated:
                    sol.evaluate()


# --- From RLOrchestrator/core/context.py ---

IntRangeSpec = int | Tuple[int, int]
Phase = Literal["exploration", "exploitation", "termination"]

@dataclass
class StageBinding:
    name: Phase
    solver: SearchAlgorithm

@dataclass
class BudgetSpec:
    max_decision_steps: IntRangeSpec
    search_steps_per_decision: IntRangeSpec
    max_search_steps: Optional[int] = None

@dataclass
class OrchestratorContext:
    problem: ProblemInterface
    stages: List[StageBinding]
    budget: BudgetSpec
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    phase_index: int = 0
    decision_count: int = 0
    search_step_count: int = 0
    max_decision_steps: int = 0
    search_steps_per_decision: int = 0
    max_search_steps: Optional[int] = None
    best_solution: Optional[Solution] = None

    def current_stage(self) -> StageBinding:
        return self.stages[self.phase_index]

    def current_solver(self) -> SearchAlgorithm:
        return self.current_stage().solver

    def current_phase(self) -> Phase:
        return self.current_stage().name

    def reset_state(self):
        self.phase_index = 0
        self.decision_count = 0
        self.search_step_count = 0
        self.best_solution = None

def normalize_range(spec: IntRangeSpec) -> Tuple[int, int]:
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        lo, hi = int(spec[0]), int(spec[1])
    else:
        value = max(1, int(spec))
        return (value, value)
    lo, hi = sorted((lo, hi))
    return (max(1, lo), max(lo, hi))

def sample_from_range(spec: Tuple[int, int], rng: np.random.Generator) -> int:
    lo, hi = spec
    return int(rng.integers(lo, hi + 1)) if lo != hi else lo

# --- From RLOrchestrator/core/stage_controller.py ---

@dataclass
class StageStepResult:
    terminated: bool
    truncated: bool
    switched: bool
    evals_run: int

class StageController:
    def __init__(self, context: OrchestratorContext):
        self.ctx = context
        self._max_decision_spec = normalize_range(context.budget.max_decision_steps)
        self._search_step_spec = normalize_range(context.budget.search_steps_per_decision)
        self._max_search_steps = context.budget.max_search_steps

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.ctx.rng = np.random.default_rng(seed)
        self.ctx.reset_state()
        self.ctx.max_decision_steps = sample_from_range(self._max_decision_spec, self.ctx.rng)
        self.ctx.search_steps_per_decision = sample_from_range(self._search_step_spec, self.ctx.rng)
        
        if hasattr(self.ctx.problem, "regenerate_instance"):
            self.ctx.problem.regenerate_instance()

        # Only initialize stages that have solvers (termination phase has None)
        for binding in self.ctx.stages:
            if binding.solver is not None:
                binding.solver.initialize()
        self._update_best()

    def step(self, action: int) -> StageStepResult:
        """Execute one decision step. Action 0 = Stay, Action 1 = Advance to next phase."""
        switched = False
        if action == 1:  # ADVANCE to next phase
            switched = self._advance_stage()

        evals_run = 0
        
        # Check if we've entered termination phase (solver is None)
        current_stage = self.ctx.stages[self.ctx.phase_index]
        in_termination = current_stage.solver is None
        
        # Only run search steps if not in termination phase
        if not in_termination:
            solver = current_stage.solver
            steps_to_run = self.ctx.search_steps_per_decision
            for _ in range(steps_to_run):
                solver.step()
                evals_run += getattr(solver, 'population_size', 0)
        
        self.ctx.search_step_count += evals_run
        self.ctx.decision_count += 1
        self._update_best()

        # Episode terminates when:
        # 1. We enter termination phase (natural end of pipeline)
        # 2. Max search steps exceeded
        terminated = in_termination or \
                     (self._max_search_steps is not None and self.ctx.search_step_count >= self._max_search_steps)
        
        # Truncated if we run out of decision budget without terminating
        truncated = (not terminated) and (self.ctx.decision_count >= self.ctx.max_decision_steps)
        
        return StageStepResult(terminated=terminated, truncated=truncated, switched=switched, evals_run=evals_run)

    def _advance_stage(self) -> bool:
        """Advance to the next phase. Returns True if successfully advanced, False if already at final phase."""
        if self.ctx.phase_index >= len(self.ctx.stages) - 1:
            return False
        
        # Export population from current solver (if it exists)
        current_solver = self.ctx.stages[self.ctx.phase_index].solver
        seeds = current_solver.export_population() if current_solver is not None else []
        
        self.ctx.phase_index += 1
        
        # Ingest population into next solver (if it exists - termination phase has no solver)
        next_stage = self.ctx.stages[self.ctx.phase_index]
        if next_stage.solver is not None:
            next_stage.solver.ingest_population(seeds)
        
        return True
    
    def _update_best(self):
        for stage in self.ctx.stages:
            if stage.solver is None:
                continue  # Skip termination phase (no solver)
            stage_best = stage.solver.get_best()
            if stage_best:
                if self.ctx.best_solution is None or stage_best < self.ctx.best_solution:
                    self.ctx.best_solution = stage_best.copy(preserve_id=True)


# --- From RLOrchestrator/core/observation.py ---

@dataclass
class ObservationState:
    solver: SearchAlgorithm
    phase: str
    step_ratio: float
    best_solution: Optional[Solution] = None
    population: Optional[List[Solution]] = None

class ObservationComputer:
    feature_names = [
        "budget_consumed", "normalized_best_fitness",
        "improvement_binary", "stagnation", "population_diversity", "active_phase"
    ]

    def __init__(self, problem_meta: dict, *, stagnation_window: int = 10, logger: logging.Logger):
        self.logger = logger
        self.fitness_lower_bound, self.fitness_upper_bound = self._extract_bounds(problem_meta)
        self.fitness_range = max(1e-9, self.fitness_upper_bound - self.fitness_lower_bound)
        self.stagnation_window = stagnation_window
        self._cached_diversity = None
        self._diversity_cache_valid = False
        self._last_population_hash = None
        self.reset()

    def reset(self):
        self.fitness_history = deque(maxlen=self.stagnation_window)
        self._last_best_fitness = float('inf')  # Track for improvement detection
    
    def compute(self, state: ObservationState) -> np.ndarray:
        # Budget consumed (0.0 at start, 1.0 at end) - aligns with reward function
        budget_consumed = state.step_ratio
        
        # Handle NaN/inf in fitness calculation
        best_fitness = state.best_solution.fitness if state.best_solution and state.best_solution.fitness is not None else float('inf')
        
        # Safe normalization with NaN handling
        if self.fitness_range <= 1e-9 or not np.isfinite(best_fitness) or best_fitness == float('inf'):
            norm_fitness = 1.0  # Worst quality if no valid fitness
        else:
            norm_fitness = (best_fitness - self.fitness_lower_bound) / self.fitness_range
            norm_fitness = np.clip(norm_fitness, 0.0, 1.0)
            if not np.isfinite(norm_fitness):
                norm_fitness = 1.0
        
        # Improvement detection: did fitness improve since last step?
        improvement_binary = 0.0
        if np.isfinite(best_fitness) and best_fitness < self._last_best_fitness - 1e-9:
            improvement_binary = 1.0
            self._last_best_fitness = best_fitness
        elif np.isfinite(best_fitness) and self._last_best_fitness == float('inf'):
            # First valid fitness is always an "improvement"
            self._last_best_fitness = best_fitness
            improvement_binary = 1.0
        
        # Only append finite fitness values
        if np.isfinite(best_fitness):
            self.fitness_history.append(best_fitness)
        
        stagnation = self._compute_stagnation()
        
        diversity = self._compute_population_diversity(state.population or state.solver.get_population())
        
        # 3-phase encoding: 0.0=exploration, 0.5=exploitation, 1.0=termination
        phase_encoding = {"exploration": 0.0, "exploitation": 0.5, "termination": 1.0}
        active_phase = phase_encoding.get(state.phase, 0.0)
        
        # Observation order matches reward function expectations:
        # [budget_consumed, fitness_norm, improvement, stagnation, diversity, phase]
        observation = np.array([
            budget_consumed,      # 0: budget used (0→1)
            norm_fitness,         # 1: quality inverse (0=best, 1=worst) 
            improvement_binary,   # 2: improved this step? (0 or 1)
            stagnation,           # 3: search stuck? (0→1)
            diversity,            # 4: population spread (0→1)
            active_phase          # 5: phase (0.0/0.5/1.0)
        ], dtype=np.float32)
        
        # Replace any NaN/inf with safe defaults
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=0.0)
        
        return observation

    def _compute_stagnation(self) -> float:
        """
        Compute stagnation as a gradual measure of search progress.
        
        FIXED: Stagnation now grows MUCH slower to allow meaningful exploration.
        
        Key design:
        1. Requires minimum history (15 steps) before reporting high stagnation
        2. Uses exponential moving average of improvement rate
        3. Only reaches 0.7+ after sustained lack of improvement
        """
        min_history_for_stagnation = 15  # Need more samples before high stagnation
        warmup_period = 8  # Steps before any significant stagnation
        
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Filter out any non-finite values
        finite_history = [f for f in self.fitness_history if np.isfinite(f)]
        if len(finite_history) < 2:
            return 0.0
        
        history_len = len(finite_history)
        
        # During warmup, keep stagnation very low
        if history_len < warmup_period:
            # Linear ramp from 0 to 0.3 during warmup
            return 0.3 * (history_len / warmup_period)
        
        # Count improvements: how many times did fitness decrease?
        improvements = 0
        for i in range(1, len(finite_history)):
            if finite_history[i] < finite_history[i-1] - 1e-9:
                improvements += 1
        
        # Improvement rate: what fraction of steps had improvement?
        improvement_rate = improvements / (len(finite_history) - 1)
        
        # Also check overall improvement from start to now
        overall_improvement = (finite_history[0] - finite_history[-1]) / (abs(finite_history[0]) + 1e-9)
        overall_improvement = max(0, overall_improvement)  # Only positive improvement counts
        
        # Base stagnation: low improvement rate = high stagnation
        # If we're improving 30%+ of steps, stagnation is 0
        # If we're improving 0% of steps, stagnation approaches 1.0
        rate_stagnation = 1.0 - np.clip(improvement_rate * 3.33, 0.0, 1.0)
        
        # Overall improvement bonus: if we're still improving overall, reduce stagnation
        overall_bonus = np.clip(overall_improvement * 2.0, 0.0, 0.3)
        
        base_stagnation = max(0.0, rate_stagnation - overall_bonus)
        
        # Confidence scaling: ramp from 0.4 to 1.0 based on history
        if history_len < min_history_for_stagnation:
            confidence = 0.4 + 0.6 * (history_len - warmup_period) / (min_history_for_stagnation - warmup_period)
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            confidence = 1.0
        
        stagnation = base_stagnation * confidence
        
        # Ensure finite result and cap at 0.95 until very long stagnation
        if not np.isfinite(stagnation):
            return 0.0
        
        # Only allow stagnation > 0.9 if truly stuck for long time
        if history_len < min_history_for_stagnation * 2:
            stagnation = min(stagnation, 0.85)
            
        return float(stagnation)

    def _compute_population_diversity(self, population: List[Solution]) -> float:
        if not population or len(population) < 2:
            return 0.0
        
        # Filter out solutions with None representations
        valid_population = [sol for sol in population if sol and sol.representation is not None]
        if len(valid_population) < 2:
            return 0.0
        
        # Check if we can use cached diversity
        try:
            current_hash = hash(tuple(sol.representation.tobytes() if hasattr(sol.representation, 'tobytes') 
                                  else str(sol.representation).encode() for sol in valid_population))
        except (AttributeError, TypeError):
            # Fallback if hashing fails
            current_hash = hash(tuple(str(sol.representation) for sol in valid_population))
        
        if self._last_population_hash == current_hash and self._cached_diversity is not None:
            return self._cached_diversity
        
        # Calculate diversity (only if cache miss)
        try:
            # Vectorized representation extraction
            reps = np.array([np.asarray(sol.representation) for sol in valid_population])
            if reps.shape[0] < 2:
                return 0.0
                
            # Handle single-dimensional arrays
            if reps.ndim == 1:
                reps = reps.reshape(1, -1)
                
            # Vectorized normalization
            reps_min = reps.min(axis=0)
            reps_max = reps.max(axis=0)
            reps_range = reps_max - reps_min
            
            # Avoid division by zero with vectorized operations
            reps_range = np.maximum(reps_range, 1e-9)
            normalized_reps = (reps - reps_min) / reps_range
            
            # Vectorized centroid and distance calculation
            centroid = np.mean(normalized_reps, axis=0)
            distances = np.linalg.norm(normalized_reps - centroid[None, :], axis=1)
            mean_dist = np.mean(distances)
            
            # Scale to [0,1] and cache
            diversity = float(np.clip(mean_dist / (math.sqrt(reps.shape[1]) / 2 + 1e-9), 0.0, 1.0))
            
            # Ensure finite result
            if not np.isfinite(diversity):
                diversity = 0.0
                
            self._cached_diversity = diversity
            self._last_population_hash = current_hash
            
            return diversity
            
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0

    @staticmethod
    def _extract_bounds(meta: dict) -> tuple[float, float]:
        if "lower_bound" in meta and "upper_bound" in meta:
            return float(meta["lower_bound"]), float(meta["upper_bound"])
        return 0.0, 1.0


# --- From RLOrchestrator/core/orchestrator.py ---

class OrchestratorEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        problem: ProblemInterface,
        exploration_solver: SearchAlgorithm,
        exploitation_solver: SearchAlgorithm,
        max_decision_steps: IntRangeSpec = 100,
        *,
        search_steps_per_decision: IntRangeSpec = 1,
        max_search_steps: Optional[int] = None,
        log_type: str = 'train',
    ):
        super().__init__()
        self.problem = problem
        self.logger = setup_logging(log_type, type(problem).__name__.lower(), 'logs', None)
        
        meta = self.problem.get_problem_info()
        self.obs_comp = ObservationComputer(meta, logger=self.logger)
        
        self.action_space = gym.spaces.Discrete(2)  # 0=STAY, 1=ADVANCE
        # Observation: [budget_consumed, fitness_norm, improvement, stagnation, diversity, phase]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        context = OrchestratorContext(
            problem=problem,
            stages=[
                StageBinding(name="exploration", solver=exploration_solver),
                StageBinding(name="exploitation", solver=exploitation_solver),
                StageBinding(name="termination", solver=None),  # Terminal phase - no solver
            ],
            budget=BudgetSpec(max_decision_steps, search_steps_per_decision, max_search_steps),
        )
        self._context = context
        self._controller = StageController(context)
        self._last_observation: Optional[np.ndarray] = None

    def reset(self, *, seed=None, options=None):
        self._controller.reset(seed=seed)
        self.obs_comp.reset()
        obs = self._observe()
        self._last_observation = obs.copy()
        return obs, {}

    def step(self, action: int):
        # Track phase before step
        phase_before = self._context.phase_index
        
        result = self._controller.step(action)
        
        # If phase switched, reset stagnation history for fresh start in new phase
        if result.switched:
            self.obs_comp.reset()
        
        obs = self._observe()
        self._last_observation = obs.copy()
        
        # Reward calculation is delegated to subclasses
        reward = 0.0 
        
        return obs, reward, result.terminated, result.truncated, {"evals_used": result.evals_run, "switched": result.switched}

    def _observe(self) -> np.ndarray:
        # Get current solver (may be None in termination phase)
        current_stage = self._context.current_stage()
        solver = current_stage.solver
        
        # For termination phase, use the last active solver's population or empty
        if solver is None:
            # Find the last stage with a solver to get population for diversity calc
            for stage in reversed(self._context.stages):
                if stage.solver is not None:
                    solver = stage.solver
                    break
        
        state = ObservationState(
            solver=solver,
            phase=self._context.current_phase(),
            step_ratio=self._context.decision_count / (self._context.max_decision_steps or 1),
            best_solution=self._context.best_solution,
        )
        return self.obs_comp.compute(state)

    def get_best_solution(self) -> Optional[Solution]:
        return self._context.best_solution
