# Bees Algorithm (BA) Implementation

This directory contains the implementation of the Bees Algorithm (BA) as described in `docs/BA.md`.

## Files

- `BA.py`: Main Bees Algorithm implementation
- `Problem.py`: Problem definitions and utilities for BA
- `README.md`: This documentation file

## Usage

### Basic Usage

```python
from algorithms.BA.BA import BeesAlgorithm
from algorithms.BA.Problem import ContinuousOptimizationProblem, create_ba_parameters

# Create a problem
problem = ContinuousOptimizationProblem(dimension=10, lower_bound=-5.0, upper_bound=5.0)

# Get default parameters
params = create_ba_parameters(population_size=50, problem_type='continuous')

# Create and run BA
ba = BeesAlgorithm(
    problem=problem,
    population_size=50,
    max_iterations=100,
    verbosity=1,
    **params
)

# Initialize and run
ba.initialize()
for _ in range(100):
    ba.step()

# Get best solution
best_solution = ba.get_best_solution()
print(f"Best fitness: {best_solution.fitness}")
```

### Parameters

The Bees Algorithm requires the following parameters:

- `n`: Total scout bees (population size)
- `m`: Number of best sites selected for neighborhood search
- `e`: Elite sites among the selected sites
- `nep`: Recruited bees per elite site
- `nsp`: Recruited bees per non-elite selected site
- `ngh`: Neighborhood radius (for local search)
- `MaxIter`: Maximum number of iterations
- `stlim` (optional): Stagnation limit to abandon a site

### Problem Types

The implementation supports both continuous and discrete problems:

- **Continuous**: Uses additive perturbations within ngh radius
- **Discrete**: Uses swap-based neighborhood operations

### Features

- ✅ Scout bee initialization
- ✅ Elite and non-elite site neighborhood search
- ✅ Global search for remaining bees
- ✅ Optional stagnation-based abandonment
- ✅ Support for continuous and discrete problems
- ✅ Integration with the project's core framework

### Enhanced Exploration Features

The BA implementation includes advanced exploration capabilities:

- **Adaptive Exploration**: Parameters automatically adjust based on iteration progress
  - Early phase (0-30%): High exploration with boosted parameters
  - Mid phase (30-70%): Balanced exploration/exploitation
  - Late phase (70%+): Focus on exploitation

- **Multiple Neighbor Strategies**:
  - Standard neighbors (within ngh radius)
  - Distant neighbors (2x ngh radius)
  - Random neighbors (completely new directions)

- **Perturbation Mechanisms**:
  - Periodic population perturbations to maintain diversity
  - Stagnation detection and recovery
  - Enhanced global search component

- **Exploration Boost**: Configurable multiplier for exploration enhancement
- **Diverse Local Search**: Multiple strategies for generating neighbors around elite sites

### Enhanced Exploration Usage

```python
# Create BA with enhanced exploration
ba = BeesAlgorithm(
    problem=problem,
    population_size=50,
    max_iterations=100,
    verbosity=1,
    exploration_boost=2.0,      # 2x exploration enhancement
    adaptive_exploration=True,  # Adaptive exploration over time
    **params
)

# The algorithm will automatically:
# - Increase exploration in early iterations (first 30%)
# - Gradually shift to exploitation (30-70%)
# - Focus on exploitation in late iterations (70%+)
# - Apply periodic perturbations to maintain diversity
# - Use multiple neighbor generation strategies
```

## Algorithm Overview

1. Initialize `n` scouts randomly within bounds
2. For each iteration:
   - Sort scouts by fitness (best first)
   - Select top `m` sites
   - Mark top `e` as elite
   - Generate `nep` neighbors for each elite site
   - Generate `nsp` neighbors for each non-elite site
   - Reinitialize remaining `(n-m)` bees randomly
   - Optional: Abandon stagnant sites

## Testing

Run the test script to verify the implementation:

```bash
python test_ba.py
```

The test includes:
- Continuous optimization on Sphere function
- Stagnation limit functionality
- Convergence verification