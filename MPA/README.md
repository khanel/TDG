# Marine Predators Algorithm (MPA)

Implements the recent predator-inspired optimizer described in
`docs/candidate_MPA.md`.

## Features

- Three-stage motion: Brownian search, transition, and Lévy-driven exploitation.
- Fish Aggregating Device (FAD) probability to inject random jumps.
- Requires bounded continuous domains for stable behavior.

## Usage

```python
from MPA import MarinePredatorsAlgorithm

solver = MarinePredatorsAlgorithm(problem, population_size=40)
solver.initialize()
for _ in range(300):
    solver.step()
best = solver.get_best_solution()
```
