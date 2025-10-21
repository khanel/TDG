# PSO-based TSP Solver

This directory contains the Particle Swarm Optimization implementation for solving the Traveling Salesman Problem (TSP).

## Files

- `tsp_pso_solver.py`: Main PSO TSP solver implementation
- `README.md`: This documentation file

## Overview

The PSO TSP solver adapts Particle Swarm Optimization for discrete permutation-based optimization. It uses a modified PSO approach that maintains TSP tour validity while leveraging swarm intelligence for exploration and exploitation.

## Key Features

### PSO Adaptation for TSP
- **Discrete PSO**: Adapted velocity and position updates for permutations
- **Tour Validity**: All operations maintain valid TSP tours
- **Swap-based Updates**: Position changes through controlled swaps
- **City 1 Fixed**: Starting city remains constant

### Swarm Intelligence
- **Particle Positions**: TSP tours (permutations)
- **Personal Best**: Each particle's best tour found
- **Global Best**: Swarm's best tour across all particles
- **Velocity Influence**: Cognitive and social components guide updates

## Usage

### Basic Usage

```python
from TSP.TSP import TSPProblem, Graph
from TSP.solvers.PSO.tsp_pso_solver import TSPPSOSolver
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

# Create and run PSO solver
solver = TSPPSOSolver(
    tsp_problem=tsp_problem,
    population_size=30,
    omega=0.7,
    c1=1.5,
    c2=1.5,
    max_iterations=200,
    verbosity=1
)

best_solution, best_fitness = solver.solve()
print(f"Best tour length: {best_fitness:.2f}")
print(f"Best route: {' -> '.join(map(str, best_solution.representation))}")
```

### Constriction Factor Usage

```python
# Create PSO with constriction factor for guaranteed convergence
solver = TSPPSOSolver(
    tsp_problem=tsp_problem,
    population_size=30,
    c1=2.05,  # c1 + c2 > 4 for constriction
    c2=2.05,
    use_constriction=True,  # Automatically calculates chi
    max_iterations=200,
    verbosity=1
)
```

### Enhanced Exploration Usage

```python
# Create PSO with enhanced exploration
solver = TSPPSOSolver(
    tsp_problem=tsp_problem,
    population_size=30,
    omega=0.7,
    c1=1.5,
    c2=1.5,
    max_iterations=200,
    verbosity=1,
    exploration_boost=1.5,      # 1.5x exploration enhancement
    adaptive_exploration=True   # Adaptive exploration over time
)

# Features:
# - Early iterations: High exploration (boosted parameters)
# - Mid iterations: Balanced exploration/exploitation
# - Late iterations: Focus on exploitation
# - Adaptive parameter adjustment
```

## Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `population_size` | Number of particles | 30 | 20-50 |
| `omega` | Inertia weight | 0.7 | 0.4-0.9 |
| `c1` | Cognitive coefficient | 1.5 | 1.0-2.5 |
| `c2` | Social coefficient | 1.5 | 1.0-2.5 |
| `max_iterations` | Maximum iterations | 200 | 100-500 |
| `use_constriction` | Use constriction factor | False | True for guaranteed convergence |
| `verbosity` | Output level | 1 | 0-1 |
| `exploration_boost` | Exploration multiplier | 1.5 | 1.0-2.0 |
| `adaptive_exploration` | Adaptive parameters | True | True for dynamic adjustment |

## PSO Variants

### 1. Inertia Weight PSO
Classical formulation with inertia weight balancing exploration/exploitation.

### 2. Constriction Factor PSO
Guaranteed convergence with constriction factor χ derived from c1 + c2 > 4.

### 3. Adaptive PSO
Parameters automatically adjust based on iteration progress for optimal performance.

## TSP-Specific Operations

### Position Updates
- **Swap Operations**: Exchange cities to move toward personal/global best
- **Probability-based**: Cognitive and social influences determine update probability
- **Tour Preservation**: City 1 always remains at tour start

### Velocity Concept
- **Discrete Velocity**: Not physical velocity, but influence strength
- **Cognitive Component**: Tendency to return to personal best
- **Social Component**: Tendency to move toward global best
- **Inertia**: Resistance to change from current position

## Algorithm Flow

1. **Initialization**:
   - Generate random TSP tours for each particle
   - Set personal best = current position
   - Find global best across swarm

2. **Iteration Loop**:
   - For each particle:
     - Calculate cognitive influence (personal best)
     - Calculate social influence (global best)
     - Apply position update with probability
     - Update personal best if improved
   - Update global best
   - Adapt exploration parameters (if enabled)

3. **Termination**: Return best solution found

## Performance Characteristics

### Exploration vs Exploitation
- **Early Phase**: High exploration with diverse position updates
- **Mid Phase**: Balanced search with both local and global influences
- **Late Phase**: Exploitation focus on refining best solutions

### Convergence Behavior
- **Stable Convergence**: Constriction factor ensures convergence
- **Adaptive Speed**: Parameters adjust for optimal convergence rate
- **Solution Quality**: Swarm intelligence finds high-quality solutions

## Visualization

When `verbosity >= 1`, the solver provides:
- **Live Plot**: Real-time visualization of current best route
- **Iteration Progress**: Fitness improvement tracking
- **Convergence Curve**: Best fitness over iterations
- **Route Visualization**: City connections and tour path

## Example Output

```
Starting TSP solution with Particle Swarm Optimization...
Number of cities: 20
Population size: 30
Max iterations: 200
Omega: 0.7, C1: 1.5, C2: 1.5
Constriction: False

Initial best fitness: 127.62
Iteration 50/200, Best fitness: 96.06
Iteration 100/200, Best fitness: 96.06
Iteration 150/200, Best fitness: 96.05
Iteration 200/200, Best fitness: 95.47

Optimization complete!
Time taken: 2.63 seconds
Best fitness: 95.47
Route: 1 -> 14 -> 18 -> 19 -> 15 -> 13 -> 9 -> 10 -> 8 -> 7 -> 6 -> 11 -> 12 -> 5 -> 4 -> 3 -> 16 -> 17 -> 20 -> 2 -> 1
```

## Comparison with Other TSP Solvers

| Algorithm | Exploration | Exploitation | TSP-Specific |
|-----------|-------------|---------------|--------------|
| PSO | Swarm diversity + social learning | Personal/global best following | Swap-based updates |
| BA | Multiple neighborhood strategies | Elite site intensification | Diverse mutations |
| GA | Crossover recombination | Tournament selection | Order crossover |
| GWO | Position-based updates | Alpha-beta-delta hierarchy | Discrete adaptations |

## Advanced Features

### Adaptive Parameter Control
- **Time-based Adaptation**: Parameters change with iteration progress
- **Performance Monitoring**: Adjust based on convergence behavior
- **Multi-strategy**: Different approaches for different optimization phases

### Swarm Intelligence
- **Information Sharing**: Particles learn from successful swarm members
- **Diversity Maintenance**: Prevents premature convergence
- **Social Learning**: Collective intelligence guides search

## Tips for Best Results

1. **Population Size**: 20-50 particles for moderate TSP instances
2. **Parameter Balance**: c1 ≈ c2 for balanced personal/social influence
3. **Constriction Factor**: Use for guaranteed convergence on complex problems
4. **Adaptive Exploration**: Enable for dynamic parameter adjustment
5. **Iteration Budget**: 100-500 iterations depending on problem size

## Dependencies

- numpy: Numerical computations
- matplotlib: Visualization (optional)
- Core framework: ProblemInterface, Solution classes
- TSP framework: TSPProblem, Graph classes

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
- Clerc, M., & Kennedy, J. (2002). The particle swarm-explosion, stability, and convergence in a multidimensional complex space.
- Wang, K. P., Huang, L., Zhou, C. G., & Pang, W. (2003). Particle swarm optimization for traveling salesman problem.