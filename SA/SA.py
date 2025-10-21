# simulated_annealing.py
import math, random, copy
from typing import Callable, Optional
import numpy as np

from Core.search_algorithm import SearchAlgorithm
from Core.problem import ProblemInterface, Solution

class SimulatedAnnealing(SearchAlgorithm):
    """
    Problem-agnostic Simulated Annealing (minimization).
    - Uses problem.neighbor(sol) if available; else falls back to generic neighbors
      inferred from problem.get_problem_info() and the representation type.
    - Population semantics: run multiple independent SA chains in parallel; each step()
      performs `moves_per_temp` Metropolis moves per chain, then cools T.
    """
    def __init__(
        self,
        problem: ProblemInterface,
        population_size: int = 1,
        *,
        initial_temp: float = 1.0,
        final_temp: float = 1e-3,
        alpha: float = 0.98,            # geometric cooling T <- alpha*T
        moves_per_temp: int = 1,
        neighbor_fn: Optional[Callable[[Solution], Solution]] = None,
        rng: Optional[random.Random] = None,
        step_scale: float = 0.1         # relative perturbation scale for continuous
    ):
        super().__init__(problem, population_size)
        assert initial_temp > 0 and final_temp > 0 and alpha > 0 and alpha < 1
        self.T0 = initial_temp
        self.Tf = final_temp
        self.alpha = alpha
        self.moves_per_temp = max(1, int(moves_per_temp))
        self._user_neighbor = neighbor_fn
        self.rng = rng or random.Random()
        self.step_scale = step_scale
        self.T = self.T0

        # Cache problem info for generic neighborhoods
        try:
            self._pinfo = self.problem.get_problem_info() or {}
        except Exception:
            self._pinfo = {}

    def initialize(self):
        super().initialize()
        self.T = self.T0

    def _accept(self, delta: float) -> bool:
        # Downhill always accepted; uphill per Metropolis (Kirkpatrick / Metropolis).
        return delta <= 0 or self.rng.random() < math.exp(-delta / self.T)

    # --- Neighborhoods ---
    def _has_problem_neighbor(self) -> bool:
        return hasattr(self.problem, "neighbor") and callable(getattr(self.problem, "neighbor"))

    def _generic_neighbor_representation(self, rep):
        """Create a perturbed copy of 'rep' based on type and problem info."""
        lb = np.array(self._pinfo.get("lower_bounds", []), dtype=float) if "lower_bounds" in self._pinfo else None
        ub = np.array(self._pinfo.get("upper_bounds", []), dtype=float) if "upper_bounds" in self._pinfo else None
        ptype = self._pinfo.get("problem_type", None)

        # numpy array -> treat as vector (continuous by default)
        if isinstance(rep, np.ndarray):
            new = rep.copy()
            if ptype == "discrete" or self._pinfo.get("domain") == "permutation":
                # Fallback: swap two positions (Černý for permutations)
                i, j = self.rng.randrange(len(new)), self.rng.randrange(len(new))
                new[i], new[j] = new[j], new[i]
                return new
            # continuous: Gaussian nudge and clamp to bounds if provided
            sigma_vec = self.step_scale * (ub - lb) if (lb is not None and ub is not None and len(ub) == len(new)) else None
            if sigma_vec is None:
                sigma = self.step_scale or 0.1
                noise = np.array([self.rng.gauss(0.0, sigma) for _ in range(len(new))])
            else:
                noise = np.array([self.rng.gauss(0.0, max(1e-12, s)) for s in sigma_vec])
            new = new + noise
            if lb is not None and ub is not None and len(ub) == len(new):
                new = np.minimum(ub, np.maximum(lb, new))
            return new

        # Python list
        if isinstance(rep, list):
            new = copy.deepcopy(rep)
            # Heuristic: if elements are unique ints -> permutation swap
            if all(isinstance(x, int) for x in new) and len(set(new)) == len(new):
                i, j = self.rng.randrange(len(new)), self.rng.randrange(len(new))
                new[i], new[j] = new[j], new[i]
                return new
            # Otherwise: pick index and perturb
            idx = self.rng.randrange(len(new))
            x = new[idx]
            if isinstance(x, (int, np.integer)):
                new[idx] = x + self.rng.choice([-1, 1])
            elif isinstance(x, (float, np.floating)):
                step = self.step_scale if not (lb is not None and ub is not None) else \
                       self.step_scale * float((ub[idx] - lb[idx]))
                new[idx] = x + self.rng.gauss(0.0, step or 0.1)
                if lb is not None and ub is not None:
                    new[idx] = min(max(new[idx], float(lb[idx])), float(ub[idx]))
            else:
                # Fallback: replace with itself (no-op) — better for the problem to supply neighbor()
                pass
            return new

        # Scalar (rare)
        if isinstance(rep, (int, float, np.floating, np.integer)):
            if isinstance(rep, int):
                return rep + self.rng.choice([-1, 1])
            else:
                return rep + self.rng.gauss(0.0, self.step_scale or 0.1)

        # Unknown type — return a deepcopy and hope the problem overrides neighbor()
        return copy.deepcopy(rep)

    def _propose(self, sol: Solution) -> Solution:
        # Prefer a problem-supplied neighbor if available
        if self._user_neighbor:
            cand = self._user_neighbor(sol)
            if not isinstance(cand, Solution):
                cand = Solution(cand, self.problem)
            return cand
        if self._has_problem_neighbor():
            cand = self.problem.neighbor(sol)  # type: ignore[attr-defined]
            if not isinstance(cand, Solution):
                cand = Solution(cand, self.problem)
            return cand

        # Generic neighbor
        rep_new = self._generic_neighbor_representation(sol.representation)
        return Solution(rep_new, self.problem)

    def step(self):
        # Ensure population is evaluated
        for sol in self.population:
            sol.evaluate()

        # Perform Metropolis moves for each chain in the population
        for _ in range(self.moves_per_temp):
            for idx, sol in enumerate(self.population):
                s_cur = sol
                e_cur = s_cur.fitness  # evaluated above
                cand = self._propose(s_cur)
                e_new = cand.evaluate()
                delta = e_new - e_cur
                if self._accept(delta):
                    # adopt candidate (replace in population)
                    self.population[idx] = cand
                # else reject; keep current
            # end for each sol
        # end for m moves

        # Cool down and update best
        self.T = max(self.Tf, self.alpha * self.T)
        self._update_best_solution()
        self.iteration += 1

    # Optional: convergence helper
    def is_cooled(self) -> bool:
        return self.T <= self.Tf
