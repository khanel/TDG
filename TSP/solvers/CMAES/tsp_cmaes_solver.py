import numpy as np
from typing import Tuple
from QD_CMA_MAE_CVT import (
    CVTArchive,
    CMAEmitter,
    ImprovementEmitter,
    RandomDirectionEmitter,
    DiscretePermutationEmitter,
    CMAMAEController,
)
from Problem_CMA_MAE import TSPProblem

def run_tsp(seed: int = 0, n_cities: int = 50, steps: int = 2000) -> Tuple[CMAMAEController, CVTArchive]:
    rng = np.random.default_rng(seed)
    # Generate TSP instance
    coords = rng.random((n_cities, 2))
    problem = TSPProblem(coords)

    # Behavior bounds (for mean/std edge length normalized): both in [0, 1.5] (generous)
    bd_bounds = [(0.0, 1.5), (0.0, 1.5)]

    # CVT archive
    archive = CVTArchive(K=256, bd_bounds=bd_bounds, alpha=0.15, samples=10000, kmeans_iters=10, rng=rng)

    # Emitters
    # 1) Continuous CMA-MAE emitter in random-keys latent
    lb, ub = problem.bounds_decode_latent()
    x0 = rng.uniform(lb, ub)
    cont_emitter = ImprovementEmitter(x0=x0, sigma=0.3, lam=32, bounds=(lb, ub), rng=rng)

    # 2) Random-direction emitter (uses same CMA engine; controller defines reward)
    dir_emitter = RandomDirectionEmitter(x0=x0, sigma=0.3, lam=32, bounds=(lb, ub), rng=rng)

    # 3) Discrete permutation emitter (swap mutation)
    init_perm = problem.get_initial_representation()
    disc_emitter = DiscretePermutationEmitter(n_items=problem.n, init_perm=init_perm, rng=rng, swap_rate=0.15)

    emitters = [cont_emitter, dir_emitter, disc_emitter]

    # Controller
    def fitness_fn(repr_obj: np.ndarray) -> float:
        return problem.fitness(repr_obj)

    def bd_fn(repr_obj: np.ndarray) -> np.ndarray:
        return problem.behavior_descriptor(repr_obj)

    def decode_fn(z: np.ndarray) -> np.ndarray:
        return problem.decode(z)

    ctrl = CMAMAEController(archive=archive, fitness_fn=fitness_fn, bd_fn=bd_fn,
                            emitters=emitters, decode_fn=decode_fn, rng=rng)

    # Simple round-robin schedule across emitters
    schedule = [0, 1, 2]  # cont -> dir -> discrete
    logs = ctrl.run(iters=steps, emitter_schedule=schedule, log_every=max(1, steps//20))

    # Print final stats
    if logs:
        last = logs[-1]
        print(f"Step {last['step']}: coverage={last['coverage']}, qd_score={last['qd_score']:.4f}")

    # Extract best found tour from archive
    best_fit = np.inf
    best_perm = None
    for cell in archive.cells.values():
        if cell['fitness'] < best_fit:
            best_fit = cell['fitness']
            best_perm = cell['repr']

    if best_perm is not None:
        print(f"Best tour length: {best_fit:.4f}")
    else:
        print("No solutions inserted.")

    return ctrl, archive

if __name__ == '__main__':
    run_tsp()
