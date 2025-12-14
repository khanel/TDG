# Artificial Bee Colony (ABC)

Implementation of the Artificial Bee Colony metaheuristic inspired by the
original formulation by Karaboga (2005) and the canonical MathWorks example
("Artificial Bee Colony (ABC) Optimization Algorithm", Global Optimization
Toolbox documentation).

## Highlights

- Employed, onlooker, and scout phases implemented with greedy selection.
- Supports continuous, binary, and permutation-based problems.
- Automatically infers `limit` thresholds when not provided (scaled by
  population size and problem dimension).
- Roulette-wheel onlooker selection with robustness for negative fitnesses.
- Population ingestion hook keeps trial counters aligned with orchestrator
  transitions.

## Usage

```python
from algorithms.ABC import ArtificialBeeColony
from algorithms.ABC.Problem import ContinuousOptimizationProblem, create_abc_parameters

problem = ContinuousOptimizationProblem(dimension=10, lower_bound=-5, upper_bound=5)
params = create_abc_parameters(population_size=40, dimension=10)

abc = ArtificialBeeColony(problem, population_size=40, **params)
abc.initialize()
for _ in range(200):
    abc.step()

best = abc.get_best_solution()
print(best.fitness)
```
