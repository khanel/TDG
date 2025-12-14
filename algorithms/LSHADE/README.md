# L-SHADE

Implementation of Success-History based Adaptive Differential Evolution with
linear population size reduction as described by Tanabe & Fukunaga (2014) and
summarized in `docs/candidate_L-SHADE.md`.

## Highlights

- Adaptive F and CR parameters sampled from historical memories.
- External archive and `p-best` mutation for diversity.
- Automatic population size reduction from the initial size to a minimum of 4.

## Usage

```python
from algorithms.LSHADE import LSHADE

solver = LSHADE(problem, population_size=80)
solver.initialize()
for _ in range(500):
    solver.step()
best = solver.get_best_solution()
```
