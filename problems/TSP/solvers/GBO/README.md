# GBO-based TSP Solver

This directory contains the Gradient-Based Optimizer implementation for solving the Traveling Salesman Problem (TSP).

## Files

- `tsp_gbo_solver.py`: Main GBO TSP solver implementation
- `README.md`: This documentation file

## Overview

The GBO TSP solver adapts Gradient-Based Optimizer for discrete permutation-based optimization. It uses GSR (Gradient Search Rule) and LEO (Local Escaping Operator) adapted for TSP tours, providing a unique approach to combinatorial optimization.

## Key Features

### GBO Adaptation for TSP
- **Discrete GSR**: Adapted gradient-inspired search for permutations
- **TSP-specific LEO**: Local escaping operations designed for tours
- **Tour validity**: All operations maintain valid TSP solutions
- **Population-based**: Multiple agents explore tour space simultaneously

### GSR (Gradient Search Rule)
- **Population-informed**: Uses reference agents to guide search direction
- **Global best influence**: Attracted to best known solution
- **Swap-based updates**: Discrete position changes through controlled swaps

### LEO (Local Escaping Operator)
Multiple escape strategies for TSP:
- **Segment reversal**: Reverse subsections of the tour
- **City relocation**: Move cities to different positions
- **Segment swapping**: Exchange tour segments between positions

## Usage

### Basic Usage

```python
from problems.TSP.TSP import TSPProblem, Graph
from problems.TSP.solvers.GBO.tsp_gbo_solver import TSPGBOSolver
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

# Create and run GBO solver
solver = TSPGBOSolver(
    tsp_problem=tsp_problem,
    population_size=30,
    alpha=1.0,
    beta=1.0,
    leo_prob=0.1,
    max_iterations=200,
    verbosity=1
)

best_solution, best_fitness = solver.solve()
print(f"Best tour length: {best_fitness:.2f}")
print(f"Best route: {' -> '.join(map(str, best_solution.representation))}")
```

### Enhanced Exploration Usage

```python
# Create GBO solver with enhanced exploration
solver = TSPGBOSolver(
    tsp_problem=tsp_problem,
    population_size=30,
    alpha=1.0,
    beta=1.0,
    leo_prob=0.1,
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
# - Multiple escape strategies
```

## Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `population_size` | Number of agents | 30 | 20-50 |
| `alpha` | Global best influence | 1.0 | 0.5-2.0 |
| `beta` | Population influence | 1.0 | 0.5-2.0 |
| `leo_prob` | Local escaping probability | 0.1 | 0.05-0.3 |
| `max_iterations` | Maximum iterations | 200 | 100-500 |
| `step_size` | GSR movement intensity | 1.0 | 0.5-2.0 |
| `verbosity` | Output level | 1 | 0-1 |
| `exploration_boost` | Exploration multiplier | 1.5 | 1.0-2.0 |
| `adaptive_exploration` | Adaptive parameters | True | True for dynamic adjustment |

## Algorithm Flow

1. **Initialization**:
   - Generate random TSP tours for each agent
   - Set personal best = current position
   - Find initial global best

2. **Iteration Loop**:
   - For each agent:
     - Select reference agents from population
     - Apply GSR to generate new tour candidate
     - Apply LEO with probability leo_prob
     - Evaluate candidate solution
     - Update personal best if improved
   - Update global best across all agents
   - Adapt exploration parameters (if enabled)

3. **Termination**: Return best solution found

## GSR Implementation for TSP

### Discrete Position Updates
- **Reference Selection**: Choose different agents as references
- **Difference Calculation**: Identify positions that differ from global best
- **Swap Operations**: Move cities toward correct positions
- **Step Size Control**: Control intensity of position changes

### Search Direction
```
D_i = α × (g - x_i) + β × (r1 - r2)
```
Adapted for discrete space using swap-based movements.

## LEO Strategies for TSP

### 1. Segment Reversal
- Select random segment of the tour
- Reverse the order of cities in that segment
- Maintains tour validity while creating new arrangements

### 2. City Relocation
- Select a city and its current position
- Choose new position in the tour
- Move city to new location, shifting others
- Creates different tour structures

### 3. Segment Swapping
- Select two non-overlapping segments
- Swap their positions in the tour
- Larger structural changes for significant exploration

## Performance Characteristics

### Exploration Mechanisms
- **GSR**: Directed search toward promising regions
- **LEO**: Random perturbations to escape local optima
- **Population Diversity**: Multiple agents explore different areas
- **Adaptive Parameters**: Dynamic balance of exploration/exploitation

### Convergence Behavior
- **Early Phase**: High exploration with frequent LEO applications
- **Mid Phase**: Balanced search with moderate parameter values
- **Late Phase**: Exploitation focus with reduced LEO probability
- **Stable Convergence**: Controlled randomization prevents instability

## Visualization

When `verbosity >= 1`, the solver provides:
- **Live Plot**: Real-time visualization of current best route
- **Iteration Progress**: Fitness improvement tracking
- **Convergence Curve**: Best fitness over iterations
- **Route Visualization**: City connections and tour path

## Example Output

```
Starting TSP solution with Gradient-Based Optimizer...
Number of cities: 20
Population size: 30
Max iterations: 200
Alpha: 1.0, Beta: 1.0, LEO Prob: 0.1

Initial best fitness: 127.62
Iteration 50/200, Best fitness: 82.64
Iteration 100/200, Best fitness: 60.25
Iteration 150/200, Best fitness: 59.16
Iteration 200/200, Best fitness: 59.16

Optimization complete!
Time taken: 2.86 seconds
Best fitness: 59.16
Route: 1 -> 2 -> 3 -> 4 -> 19 -> 18 -> 20 -> 17 -> 16 -> 15 -> 14 -> 13 -> 12 -> 11 -> 10 -> 9 -> 8 -> 7 -> 5 -> 6 -> 1
```

## Comparison with Other TSP Solvers

| Algorithm | Exploration | Exploitation | TSP-Specific |
|-----------|-------------|---------------|--------------|
| GBO | GSR + multiple LEO strategies | Population-informed search | Segment ops, city relocation |
| BA | Multiple neighborhood strategies | Elite site intensification | Diverse mutations |
| PSO | Social learning | Personal/global best following | Swap-based updates |
| GA | Crossover recombination | Tournament selection | Order crossover |

## Advanced Features

### Adaptive Parameter Control
- **Time-based Adaptation**: Parameters change with iteration progress
- **Performance Monitoring**: Adjust based on convergence behavior
- **Multi-strategy LEO**: Different escape operations for different situations

### Population-based Search
- **Information Sharing**: Agents influence each other through GSR
- **Diversity Maintenance**: LEO preserves population diversity
- **Global Guidance**: Global best attracts all agents

## Tips for Best Results

1. **Population Size**: 20-50 agents for moderate TSP instances
2. **LEO Probability**: 0.1-0.2 for balanced exploration/exploitation
3. **Alpha/Beta Balance**: Equal values (1.0) for symmetric influence
4. **Adaptive Exploration**: Enable for dynamic parameter adjustment
5. **Iteration Budget**: 100-300 iterations depending on problem size

## Dependencies

- numpy: Numerical computations and random operations
- matplotlib: Visualization (optional)
- Core framework: ProblemInterface, Solution classes
- TSP framework: TSPProblem, Graph classes

## References

- Ahmadianfar, I., Bozorg-Haddad, O., & Chu, X. (2020). Gradient-based optimizer: A new metaheuristic optimization algorithm.
- Ahmadianfar, I., et al. (2021). GBO applications to various optimization problems.
- Survey papers on population-based metaheuristics and GBO performance analysis.

## Algorithm Variants

### Standard GBO
Classical implementation with fixed parameters.

### Adaptive GBO
Parameters adjust based on iteration progress.

### Enhanced GBO
Multiple LEO strategies and improved exploration.

### Discrete GBO (TSP)
Adapted for permutation-based optimization with tour-specific operations.