# Whale Optimization Algorithm (WOA)

Implementation of the bubble-net feeding strategy in
`docs/candidate_WOA.md`.

## Highlights

- Encircling prey, search for prey, and spiral bubble-net operations.
- Time-varying coefficient `a` transitions from exploration to exploitation.
- Requires bounded domains for stable movement.

## Usage

```python
from algorithms.WOA import WhaleOptimizationAlgorithm

solver = WhaleOptimizationAlgorithm(problem, population_size=30)
solver.initialize()
for _ in range(300):
    solver.step()
print(solver.get_best_solution().fitness)
```
