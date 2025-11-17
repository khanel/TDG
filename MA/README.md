# Memetic Algorithm (MA)

Implements the hybrid GA + local search approach described in
`docs/candidate_MA.md`.

## Key ideas

- Tournament selection, uniform crossover, and mutation.
- Local-improvement memes triggered probabilistically per offspring.
- Can leverage `ProblemInterface.sample_neighbors` for domain-specific refinements.

## Usage

```python
from MA import MemeticAlgorithm

solver = MemeticAlgorithm(problem, population_size=40)
solver.initialize()
for _ in range(200):
    solver.step()
print(solver.get_best_solution().fitness)
```
