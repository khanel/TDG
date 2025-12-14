# Harris Hawks Optimization (HHO)

Implementation inspired by the 2019 HHO publication and
`docs/candidate_HHO.md`.

## Characteristics

- Explicit exploration/exploitation split controlled by escaping energy `E`.
- Soft/hard besiege strategies and rapid dives powered by Lévy flights.
- Works with any `ProblemInterface` exposing bounds for clipping.

## Usage

```python
from algorithms.HHO import HarrisHawksOptimization

solver = HarrisHawksOptimization(problem, population_size=50)
solver.initialize()
for _ in range(300):
    solver.step()
print(solver.get_best_solution().fitness)
```
