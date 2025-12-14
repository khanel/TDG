# NS-based TSP Solver

This directory contains the Novelty Search implementation for solving the Traveling Salesman Problem (TSP).

## Files

- `tsp_ns_solver.py`: Main NS TSP solver implementation
- `README.md`: This documentation file

## Overview

The NS TSP solver adapts Novelty Search for discrete permutation-based optimization. Instead of optimizing for tour length, it optimizes for behavioral novelty, discovering diverse and unexpected tour patterns that traditional optimization might miss.

## Key Features

### NS Adaptation for TSP
- **Behavioral Novelty**: Selection based on tour pattern novelty, not length
- **Archive-based Memory**: Remembers previously discovered novel tour patterns
- **Diverse Exploration**: Discovers multiple solution strategies
- **TSP-specific Behaviors**: Characterizes tours by structure, edges, and patterns

### Behavior Characterization
- **Edge Usage Patterns**: Which city pairs are used in tours
- **City Visit Sequences**: Order and clustering of city visits
- **Tour Structure Features**: Path length, connectivity, movement patterns
- **Diversity Metrics**: Coverage, repetition, and distribution measures

### Reproduction Operators
- **Swap Operations**: Exchange cities to create new patterns
- **Insert Operations**: Move cities to different positions
- **Reverse Operations**: Invert tour segments
- **Scramble Operations**: Randomize tour subsections

## Usage

### Basic Usage

```python
from problems.TSP.TSP import TSPProblem, Graph
from problems.TSP.solvers.NS.tsp_ns_solver import TSPNSSolver
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

# Create and run NS solver
solver = TSPNSSolver(
    tsp_problem=tsp_problem,
    population_size=50,
    k_neighbors=15,
    archive_threshold=0.0,  # Archive all novel behaviors
    max_archive_size=500,   # Limit archive size
    max_iterations=200,
    verbosity=1
)

best_solution, best_fitness = solver.solve()
print(f"Best tour length: {best_fitness:.2f}")
print(f"Best route: {' -> '.join(map(str, best_solution.representation))}")
```

### Advanced Configuration

```python
# Configure for high exploration
solver = TSPNSSolver(
    tsp_problem=tsp_problem,
    population_size=100,      # Larger population for diversity
    k_neighbors=10,           # Fewer neighbors for local novelty
    archive_threshold=0.0,    # Archive everything initially
    max_archive_size=1000,    # Large archive for memory
    max_iterations=500,       # More iterations for exploration
    verbosity=1
)

# Configure for focused diversity
solver = TSPNSSolver(
    tsp_problem=tsp_problem,
    population_size=30,       # Smaller population
    k_neighbors=25,           # More neighbors for broader assessment
    archive_threshold=0.1,    # Selective archiving
    max_archive_size=200,     # Smaller archive
    max_iterations=200,
    verbosity=1
)
```

## Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `population_size` | Number of individuals | 50 | 30-100 |
| `k_neighbors` | Neighbors for novelty computation | 15 | 5-25 |
| `archive_threshold` | Threshold for archiving | 0.0 | 0.0-0.5 |
| `max_archive_size` | Maximum archive size | None | 200-1000 |
| `max_iterations` | Maximum iterations | 200 | 100-500 |
| `verbosity` | Output level | 1 | 0-1 |

## Algorithm Flow

1. **Initialization**:
   - Generate random TSP tours
   - Initialize empty archive
   - Characterize initial behaviors

2. **Generation Loop**:
   - Characterize behaviors of all individuals
   - Compute novelty scores using archive
   - Select parents based on novelty (not fitness)
   - Apply TSP-specific reproduction operators
   - Update archive with novel behaviors
   - Replace population with offspring

3. **Termination**: Return individual with highest novelty score

## Novelty Computation

### TSP Behavior Characterization
```python
behavior = [
    unique_edges / n,           # Edge diversity
    repeated_edges / n,         # Edge repetition
    avg_degree / 2.0,          # Average city connections
    max_degree / n,            # Maximum city connections
    total_distance / 1000.0,   # Normalized tour length
    len(set(tour)) / n,        # City coverage
    np.std(city_degrees) / 2.0, # Degree variance
    unique_ratio,              # Unique edge ratio
    n / 50.0,                  # Problem size factor
    jump_size / 25.0           # City jump size
]
```

### Novelty Score Calculation
```
novelty_score = average_distance_to_k_nearest_neighbors(behavior, archive)
```

### Archive Dynamics
- **Growth Phase**: Archive expands rapidly with novel discoveries
- **Saturation Phase**: Archive reaches capacity, replaces oldest entries
- **Maintenance Phase**: Selective archiving of highly novel behaviors

## TSP-Specific Operations

### Reproduction Operators
- **Swap**: `tour[i], tour[j] = tour[j], tour[i]`
- **Insert**: Move city to different position
- **Reverse**: Invert segment between positions
- **Scramble**: Randomize segment order

### Behavior Features
- **Edge-based**: Which city pairs are connected
- **Sequence-based**: Order of city visits
- **Structural**: Path length, connectivity patterns
- **Movement-based**: Direction and distance patterns

## Performance Characteristics

### Exploration vs Exploitation
- **Pure Exploration**: No fitness bias in selection
- **Behavioral Diversity**: Focus on discovering new patterns
- **Archive Memory**: Remembers discovered novelty
- **Open-ended Discovery**: Can find unexpected solutions

### Convergence Behavior
- **Archive Growth**: Rapid initial expansion
- **Novelty Saturation**: Eventually reaches stable novelty levels
- **Diverse Solutions**: Maintains population diversity
- **Pattern Discovery**: Finds multiple solution strategies

## Visualization

When `verbosity >= 1`, the solver provides:
- **Live Plot**: Real-time visualization of current best route
- **Iteration Progress**: Archive size and novelty tracking
- **Convergence Curves**: Fitness and novelty evolution
- **Behavioral Metrics**: Diversity and uniqueness measures

## Example Output

```
Starting TSP solution with Novelty Search...
Number of cities: 20
Population size: 50
Max iterations: 200
k-neighbors: 15, Archive threshold: 0.0

Initial archive size: 50
Iteration 50/200
  Archive size: 500
  Avg novelty: 0.0066
  Best fitness: 158.79
  Behavioral diversity: 0.0524

Iteration 100/200
  Archive size: 500
  Avg novelty: 0.0057
  Best fitness: 124.67
  Behavioral diversity: 0.0459

Iteration 200/200
  Archive size: 500
  Avg novelty: 0.0065
  Best fitness: 137.22
  Behavioral diversity: 0.0617

Optimization complete!
Time taken: 24.31 seconds
Final archive size: 500
Best fitness: 137.22

Behavioral Diversity Metrics:
  Behavioral diversity: 0.0617
  Unique behaviors: 50
```

## Key Differences from Traditional TSP Solvers

| Aspect | Traditional TSP Solvers | Novelty Search TSP |
|--------|------------------------|-------------------|
| **Selection** | Based on tour length | Based on behavioral novelty |
| **Goal** | Shortest possible tour | Diverse tour patterns |
| **Archive** | Not used | Stores novel behaviors |
| **Convergence** | To optimal tour | To diverse tour set |
| **Discovery** | Local improvements | Novel solution strategies |

## Advanced Features

### Archive Management
- **Size Control**: Prevents unlimited growth
- **Quality Filtering**: Selective archiving of novel behaviors
- **Memory Efficiency**: FIFO replacement for old behaviors

### Diversity Maintenance
- **Behavioral Metrics**: Quantifies population diversity
- **Novelty Distribution**: Tracks range of novelty scores
- **Exploration Balance**: Maintains exploration throughout run

### Reproduction Strategies
- **Operator Selection**: Multiple mutation operators
- **Adaptive Intensity**: Adjusts mutation strength
- **TSP Feasibility**: Maintains valid tour structures

## Tips for Best Results

1. **Archive Size**: Larger archives (500+) for complex problems
2. **k_neighbors**: 10-20 for balanced local/global novelty assessment
3. **Population Size**: 50-100 for sufficient diversity
4. **Archive Threshold**: 0.0 for initial exploration, >0.0 for selectivity
5. **Iteration Budget**: 200-500 iterations for thorough exploration

## Dependencies

- numpy: Numerical computations and distance calculations
- matplotlib: Visualization (optional)
- Core framework: ProblemInterface, Solution classes
- TSP framework: TSPProblem, Graph classes

## Applications

### When to Use NS for TSP
- **Diverse Solutions**: Need variety of tour patterns
- **Exploration Focus**: Primary goal is discovering new strategies
- **Open-ended Problems**: No single optimal solution
- **Research**: Studying TSP solution space characteristics

### Complementary to Traditional Methods
- **Hybrid Approaches**: Combine NS with fitness-based methods
- **Solution Seeding**: Use NS to seed traditional optimization
- **Diversity Analysis**: Understand TSP solution space structure

## References

- Lehman, J., & Stanley, K. O. (2008). Exploiting open-endedness to solve problems through the search for novelty.
- Mouret, J. B., & Doncieux, S. (2012). Novelty-based multiobjectivization.
- Gomes, J., et al. (2015). Novelty search for the traveling salesman problem.

## Algorithm Variants

### Standard NS
Classical implementation with fixed parameters.

### Minimal Criteria NS
Combines novelty with minimum tour length requirements.

### Quality Diversity NS
Maintains both novelty and quality in different solution niches.

### Archive-based NS
Various archive management strategies (random, quality-based, etc.).