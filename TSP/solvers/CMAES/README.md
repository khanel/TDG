# TSP CMA-ES Solver

This directory contains the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) solver for the Traveling Salesman Problem (TSP).

## Overview

CMA-ES is a powerful evolutionary algorithm for continuous optimization that adapts the covariance matrix of a multivariate normal distribution. Since CMA-ES is designed for continuous optimization, we adapt it for TSP by using a continuous representation that gets converted to discrete tours.

## Files

- `tsp_cmaes_solver.py`: Main CMA-ES solver implementation for TSP
- `README.md`: This documentation file

## Key Features

### CMA-ES Algorithm Features
- **Covariance Matrix Adaptation**: Dynamically adapts the search distribution
- **Evolution Paths**: Uses p_σ and p_c for step-size and covariance adaptation
- **Step-Size Control**: Adaptive step-size control with σ updates
- **Population-Based**: Uses (μ/μ_w, λ) selection strategy

### TSP-Specific Adaptations
- **Continuous-to-Discrete Conversion**: Converts continuous CMA-ES solutions to TSP tours
- **Live Visualization**: Real-time plotting of TSP routes during optimization
- **Robust Error Handling**: Handles headless environments gracefully
- **Comprehensive Logging**: Detailed progress tracking and statistics

## Usage

### Basic Usage

```python
from TSP.TSP import TSPProblem, Graph
from TSP.solvers.CMA_ES.tsp_cmaes_solver import TSPCmaesSolver

# Create TSP problem
graph = Graph(distance_matrix)
tsp_problem = TSPProblem(graph, city_coordinates)

# Create and configure CMA-ES solver
solver = TSPCmaesSolver(
    tsp_problem=tsp_problem,
    population_size=None,  # Use default: 4 + floor(3*ln(n))
    sigma=0.5,            # Initial step size
    max_iterations=200,   # Maximum iterations
    verbosity=1           # Verbosity level
)

# Solve the problem
best_solution, best_fitness = solver.solve()

print(f"Best tour length: {best_fitness:.4f}")
print(f"Best route: {' -> '.join(map(str, best_solution.representation))}")
```

### Advanced Configuration

```python
# Custom population size
solver = TSPCmaesSolver(
    tsp_problem=tsp_problem,
    population_size=50,   # Custom λ
    sigma=0.3,           # Smaller initial step size
    max_iterations=500,  # More iterations
    verbosity=2          # More detailed output
)
```

## Algorithm Parameters

### CMA-ES Parameters
- **λ (lambda_)**: Population size (default: 4 + floor(3*ln(n)))
- **μ (mu)**: Number of parents (default: floor(λ/2))
- **σ (sigma)**: Initial step size (default: 0.5)
- **Learning Rates**:
  - `cc`: Evolution path for covariance matrix (auto-calculated)
  - `cs`: Evolution path for step-size (auto-calculated)
  - `c1`: Rank-one update learning rate (auto-calculated)
  - `cmu`: Rank-μ update learning rate (auto-calculated)
  - `damps`: Damping parameter (auto-calculated)

### TSP-Specific Parameters
- **Continuous Representation**: Uses [0,1] bounds for continuous variables
- **Tour Conversion**: Sorts continuous values to create discrete tours
- **Visualization**: Updates every 10 iterations, shows final result

## Output and Visualization

### Console Output
```
Starting TSP solution with CMA-ES...
Number of cities: 20
Population size: None
Max iterations: 200
Initial sigma: 0.5
CMA-ES initialized with λ=13, μ=6

Iteration 50/200
  Best fitness: 45.6789
  Sigma: 0.0345
  Condition number: 2.34

Iteration 100/200
  Best fitness: 42.1234
  Sigma: 0.0123
  Condition number: 1.87

Optimization complete!
Time taken: 15.67 seconds
Best fitness: 41.2345
Route: 1 -> 5 -> 12 -> 8 -> 15 -> 3 -> 10 -> 7 -> 18 -> 20 -> 14 -> 2 -> 9 -> 16 -> 11 -> 4 -> 13 -> 17 -> 19 -> 6 -> 1

CMA-ES Final State:
  Final sigma: 0.0089
  Final condition number: 1.45
  Axis ratio: 1.45
```

### Visualization
- **Live Plotting**: Shows TSP route evolution every 10 iterations
- **Convergence Plot**: Fitness, mean fitness, and step-size evolution
- **Final Route**: Best solution visualization
- **Headless Support**: Gracefully handles environments without display

## Technical Details

### CMA-ES Algorithm Flow
1. **Initialization**: Set mean vector, covariance matrix, evolution paths
2. **Population Sampling**: Sample λ individuals from N(m, σ²C)
3. **Evaluation**: Convert continuous solutions to TSP tours and evaluate
4. **Selection**: Select best μ individuals
5. **Update**: Update mean, evolution paths, covariance matrix, and step-size
6. **Repeat**: Until convergence or max iterations reached

### TSP Adaptation Strategy
- **Continuous Representation**: Each city gets a continuous value [0,1]
- **Tour Construction**: Sort cities by their continuous values
- **Permutation Generation**: Ensure valid city permutations
- **Fitness Evaluation**: Use standard TSP distance calculation

### Memory and Performance
- **Space Complexity**: O(n²) for covariance matrix (n = dimension)
- **Time Complexity**: O(λ × n²) per iteration (sampling + eigendecomposition)
- **Scalability**: Suitable for problems up to ~100 cities
- **Parallelization**: Population evaluation can be parallelized

## Comparison with Other Algorithms

### vs. Bees Algorithm (BA)
- CMA-ES has better theoretical convergence guarantees
- CMA-ES adapts covariance matrix for correlated variables
- BA may be faster for small problems but CMA-ES scales better

### vs. Particle Swarm Optimization (PSO)
- CMA-ES has more sophisticated adaptation mechanisms
- CMA-ES performs better on correlated, non-separable problems
- PSO may converge faster on simple problems

### vs. Novelty Search (NS)
- CMA-ES provides more directed optimization
- NS explores more diverse solutions
- CMA-ES typically finds better solutions for single-objective TSP

## Best Practices

### Parameter Tuning
1. **Population Size**: Start with default (4 + 3*ln(n)), increase for difficult problems
2. **Initial Sigma**: 0.5-1.0 for most problems, smaller for fine-tuning
3. **Max Iterations**: 100-500 depending on problem size and difficulty

### Problem-Specific Considerations
1. **City Distribution**: CMA-ES works well with clustered cities
2. **Distance Metrics**: Performs well with Euclidean distances
3. **Problem Size**: Best for 20-100 cities

### Performance Optimization
1. **Eigendecomposition**: Updated every ~λ/10 iterations for efficiency
2. **Bounds Handling**: Solutions are clipped to problem bounds
3. **Early Stopping**: Monitor fitness improvement for convergence

## Troubleshooting

### Common Issues
1. **Poor Convergence**: Try increasing population size or reducing sigma
2. **Memory Issues**: Reduce population size for large problems
3. **No Visualization**: Check DISPLAY environment variable
4. **Slow Performance**: Reduce verbosity or max iterations

### Error Messages
- **"μ cannot be larger than λ"**: Reduce mu or increase lambda
- **"Covariance matrix not positive definite"**: Check problem bounds
- **"Display not available"**: Visualization disabled automatically

## References

1. Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159-195.

2. Auger, A., & Hansen, N. (2005). A restart CMA evolution strategy with increasing population size. *Proceedings of the IEEE Congress on Evolutionary Computation*, 1769-1776.

3. Hansen, N. (2006). The CMA evolution strategy: A tutorial. *arXiv preprint arXiv:1604.00772*.

## Future Enhancements

- **Multi-objective CMA-ES**: For multi-objective TSP variants
- **CMA-ES with restarts**: Automatic restart mechanisms
- **Parallel CMA-ES**: Distributed population evaluation
- **Constraint handling**: For constrained TSP problems
- **Hybrid approaches**: Combine with local search methods