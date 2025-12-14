# Slime Mould Algorithm (SMA)

Implements the dynamic propagation/contraction strategy summarized in
`docs/candidate_SMA.md`.

## Highlights

- Rank-based oscillation weights approximate positive/negative feedback.
- Uses combination of guided moves toward the best solution and differential
  updates using other agents.
- Requires bounded continuous domains and works as an exploration-biased solver.

## Usage

```python
from algorithms.SMA import SlimeMouldAlgorithm

solver = SlimeMouldAlgorithm(problem, population_size=40)
solver.initialize()
for _ in range(200):
    solver.step()
best = solver.get_best_solution()
```
