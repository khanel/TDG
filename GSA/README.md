# Gravitational Search Algorithm (GSA)

Implementation based on Rashedi et al., “GSA: A Gravitational Search Algorithm”
and the internal summary in `docs/candidate_GSA.md`.

## Features

- Continuous search using Newtonian gravitational metaphor.
- Adaptive gravitational constant `G(t) = G0 * exp(-alpha * t/T)`.
- Roulette subset of k-best masses to focus exploitation.
- Bound handling via clipping using `ProblemInterface.get_problem_info()`.

## Usage

```python
from GSA import GravitationalSearchAlgorithm

solver = GravitationalSearchAlgorithm(problem, population_size=40)
solver.initialize()
for _ in range(200):
    solver.step()
best = solver.get_best_solution()
```
