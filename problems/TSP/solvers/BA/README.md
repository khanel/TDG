# Bees Algorithm (BA) TSP Solver

This directory contains the Bees Algorithm implementation for solving the Traveling Salesman Problem (TSP).

## Files

- `tsp_ba_solver.py`: Main BA TSP solver implementation
- `README.md`: This documentation file

## Overview

The Bees Algorithm TSP solver uses the Bees Algorithm with TSP-specific neighborhood operations:

- **Scout bees**: Initial population of random TSP tours
- **Elite sites**: Best solutions get more neighborhood exploration (nep bees each)
- **Non-elite sites**: Good solutions get moderate exploration (nsp bees each)
- **Global search**: Remaining bees explore new random solutions

## TSP-Specific Features

### Neighborhood Operations
- **Swap**: Exchange two cities in the tour
- **Insert**: Move a city to a different position
- **Reverse**: Reverse a subsection of the tour

### Tour Validity
- All operations maintain valid TSP tours (each city visited exactly once)
- City 1 is always fixed at the start of the tour
- Operations only affect cities 2-N to maintain feasibility

## Usage

### Basic Usage

```python
from problems.TSP.TSP import TSPProblem, Graph
from problems.TSP.solvers.BA.tsp_ba_solver import TSPBASolver
import numpy as np

# Create TSP problem
num_cities = 20
city_coords = np.random.rand(num_cities, 2) * 100

# Calculate distance matrix
distances = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            distances[i,j] = np.sqrt(np.sum((city_coords[i] - city_coords[j])**2))

graph = Graph(distances)
tsp_problem = TSPProblem(graph, city_coords)

# Create and run BA solver
solver = TSPBASolver(
    tsp_problem=tsp_problem,
    population_size=50,
    m=10,      # Select 10 best sites
    e=3,       # 3 elite sites
    nep=8,     # 8 bees per elite site
    nsp=4,     # 4 bees per non-elite site
    ngh=2,     # 2 neighborhood operations per neighbor
    max_iterations=200,
    stlim=15,  # Abandon sites after 15 iterations without improvement
    verbosity=1
)

best_solution, best_fitness = solver.solve()
print(f"Best tour length: {best_fitness:.2f}")
print(f"Best route: {' -> '.join(map(str, best_solution.representation))}")
```

### Enhanced Exploration Usage

```python
# Create BA solver with enhanced exploration
solver = TSPBASolver(
    tsp_problem=tsp_problem,
    population_size=50,
    m=10, e=3, nep=8, nsp=4, ngh=2,
    max_iterations=200,
    stlim=15,
    verbosity=1,
    exploration_boost=2.0,      # 2x exploration enhancement
    adaptive_exploration=True   # Adaptive exploration over time
)

# Features:
# - Early iterations: High exploration (boosted parameters)
# - Mid iterations: Balanced exploration/exploitation
# - Late iterations: Focus on exploitation
# - Periodic perturbations maintain diversity
# - Multiple neighbor generation strategies
```

### Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `population_size` | Number of scout bees (n) | 50 | 20-100 |
| `m` | Sites selected for neighborhood search | 10 | 5-20 |
| `e` | Elite sites among selected | 3 | 2-5 |
| `nep` | Bees per elite site | 8 | 5-15 |
| `nsp` | Bees per non-elite site | 4 | 2-8 |
| `ngh` | Neighborhood operations per neighbor | 2 | 1-3 |
| `max_iterations` | Maximum iterations | 200 | 100-500 |
| `stlim` | Stagnation limit (optional) | None | 10-20 |
| `verbosity` | Output level (0=quiet, 1=verbose) | 1 | 0-1 |

## Algorithm Flow

1. **Initialization**: Generate random TSP tours starting with city 1
2. **Site Selection**: Sort by fitness, select top m sites
3. **Elite Search**: Generate nep neighbors for top e elite sites
4. **Non-Elite Search**: Generate nsp neighbors for remaining m-e sites
5. **Global Search**: Generate random tours for remaining bees
6. **Optional Abandonment**: Replace stagnant sites with random tours
7. **Repeat** until max_iterations reached

## Performance Characteristics

- **Exploration**: Random initialization and global search
- **Exploitation**: Intensive local search around elite sites
- **Balance**: Configurable ratio of exploration vs exploitation
- **Robustness**: Stagnation handling prevents premature convergence

## Example Output

```
Starting TSP solution with Bees Algorithm...
Number of cities: 20
Population size: 50
Max iterations: 200
Selected sites (m): 10, Elite sites (e): 3
Elite bees per site (nep): 8, Non-elite bees per site (nsp): 4

Initial best fitness: 145.67
Iteration 50, Best fitness: 89.23
Iteration 100, Best fitness: 76.45
Iteration 150, Best fitness: 72.18
Iteration 200, Best fitness: 69.34

Optimization complete!
Time taken: 3.45 seconds
Best fitness: 69.34
Route: 1 -> 5 -> 3 -> 7 -> 2 -> 8 -> 4 -> 6 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20 -> 1
```

## Visualization

When `verbosity >= 1`, the solver provides:
- Live plot showing current best route
- Iteration-by-iteration progress updates
- Final convergence curve
- Route visualization with city connections

## Comparison with Other Algorithms

| Algorithm | Exploration | Exploitation | TSP-Specific Ops |
|-----------|-------------|---------------|------------------|
| BA | Global search + random neighbors | Elite site intensification | Swap, Insert, Reverse |
| GA | Crossover recombination | Tournament selection | Order crossover, mutation |
| GWO | Position updates | Alpha-beta-delta following | Discrete position updates |

## Tips for Best Results

1. **Population Size**: Larger populations (50-100) for complex problems
2. **Elite Ratio**: More elite sites (higher e/m ratio) for exploitation
3. **Neighborhood Size**: More operations (higher ngh) for local search
4. **Stagnation Limit**: Enable for long runs to prevent local optima traps
5. **Parameter Tuning**: Use smaller ngh (1-2) for early exploration, larger for exploitation

## Dependencies

- numpy
- matplotlib (for visualization)
- Core framework (ProblemInterface, Solution)
- BA framework (BeesAlgorithm)
- TSP framework (TSPProblem, Graph)