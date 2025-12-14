# Gradient-Based Optimizer (GBO) Implementation

This directory contains the implementation of Gradient-Based Optimizer (GBO) as described in `docs/GBO.md`.

## Files

- `GBO.py`: Main Gradient-Based Optimizer implementation
- `Problem.py`: Problem definitions and utilities for GBO
- `README.md`: This documentation file

## Overview

Gradient-Based Optimizer uses two core operators inspired by gradient-based optimization:

- **GSR (Gradient Search Rule)**: Guides solutions using population-informed search vectors
- **LEO (Local Escaping Operator)**: Injects controlled randomness to escape local optima

This implementation follows the canonical GBO approach introduced by Ahmadianfar et al. (2020).

## Key Features

### Core GBO Components
- **Population-based**: Multiple agents search the solution space simultaneously
- **Gradient-inspired**: Uses population information to create search directions
- **Local escaping**: Controlled randomization to avoid local optima
- **Adaptive parameters**: Coefficients adjust based on iteration progress

### Enhanced Exploration
- **Adaptive GSR/LEO**: Parameters change over time for optimal performance
- **Multiple escape strategies**: Diverse LEO operations for different scenarios
- **Population diversity**: Maintains exploration throughout optimization

## Usage

### Basic Usage

```python
from algorithms.GBO.GBO import GradientBasedOptimizer
from algorithms.GBO.Problem import ContinuousOptimizationProblem, create_gbo_parameters

# Create problem
problem = ContinuousOptimizationProblem(dimension=10, lower_bound=-5.0, upper_bound=5.0)

# Create GBO parameters
params = create_gbo_parameters(population_size=30, problem_type='continuous')

# Create and run GBO
gbo = GradientBasedOptimizer(
    problem=problem,
    population_size=30,
    max_iterations=100,
    verbosity=1,
    **params
)

# Initialize and run
gbo.initialize()
for _ in range(100):
    gbo.step()

# Get best solution
best_solution = gbo.get_best_solution()
print(f"Best fitness: {best_solution.fitness}")
```

### Enhanced Exploration Usage

```python
# Create GBO with enhanced exploration
gbo = GradientBasedOptimizer(
    problem=problem,
    population_size=30,
    max_iterations=100,
    verbosity=1,
    alpha=1.0,          # Global best influence
    beta=1.0,           # Population influence
    leo_prob=0.1,       # Local escaping probability
    step_size=1.0,      # GSR step size
    exploration_boost=1.5,      # 1.5x exploration enhancement
    adaptive_exploration=True    # Adaptive exploration over time
)

# Features:
# - Early iterations: High exploration (boosted parameters)
# - Mid iterations: Balanced exploration/exploitation
# - Late iterations: Focus on exploitation
# - Adaptive parameter adjustment
# - Multiple escape strategies
```

## Parameters

### Core Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `population_size` | Number of agents | 30 | 10-100 |
| `alpha` | Global best influence coefficient | 1.0 | 0.5-2.0 |
| `beta` | Population influence coefficient | 1.0 | 0.5-2.0 |
| `leo_prob` | Local escaping probability | 0.1 | 0.05-0.3 |
| `max_iterations` | Maximum iterations | 100 | 50-500 |
| `step_size` | GSR movement step size | 1.0 | 0.5-2.0 |

### Enhanced Exploration Parameters
| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `exploration_boost` | Exploration multiplier | 1.0 | > 1.0 increases exploration |
| `adaptive_exploration` | Adaptive parameters | False | Enables time-based adaptation |

## Algorithm Components

### Gradient Search Rule (GSR)
```
D_i = α × (g - x_i) + β × (r1 - r2)
y_i = x_i + step_size × D_i
```

Where:
- `g`: Global best position
- `r1, r2`: Random reference agents
- `D_i`: Search direction vector
- `y_i`: Proposed new position

### Local Escaping Operator (LEO)
Multiple escape strategies:
- **Gaussian noise**: Random perturbation around current position
- **Uniform perturbation**: Random movement within bounds
- **Directional escape**: Movement toward global best with random component

## Problem Types

### Continuous Problems
- Standard GSR with real-valued position updates
- Boundary handling with clipping
- LEO provides controlled randomization

### Discrete Problems (TSP)
- Adapted GSR using permutation-based operations
- Swap-based position updates
- TSP-specific LEO operations (segment reversal, city relocation)

## Parameter Adaptation

### Time-based Adaptation
- **Early Phase (0-30%)**: High exploration
  - Increased α, β, leo_prob
  - Larger step_size
- **Mid Phase (30-70%)**: Balanced search
  - Moderate parameter values
  - Stable exploration/exploitation
- **Late Phase (70%+)**: Exploitation focus
  - Reduced leo_prob
  - Smaller step_size

### Problem-specific Tuning
Different parameter sets for different benchmark problems:
- **Sphere**: Balanced parameters
- **Rastrigin**: Higher exploration
- **Ackley**: Moderate exploration
- **Griewank**: Lower exploration

## Performance Characteristics

### Exploration vs Exploitation
- **GSR**: Provides directed search toward promising regions
- **LEO**: Maintains diversity and prevents premature convergence
- **Adaptive parameters**: Balances exploration/exploitation over time

### Convergence Behavior
- **Stable convergence**: Controlled randomization prevents instability
- **Population diversity**: Multiple agents explore different regions
- **Local optima escape**: LEO helps escape suboptimal solutions

## Advanced Features

### Multiple LEO Strategies
- **Gaussian**: Smooth, local perturbations
- **Uniform**: Random exploration within bounds
- **Directional**: Guided movement with randomization

### Population-based Search
- **Information sharing**: Agents influence each other through GSR
- **Diversity maintenance**: LEO preserves population diversity
- **Global guidance**: Global best attracts all agents

## Example Problems

### Sphere Function (Unimodal)
```python
problem = ContinuousOptimizationProblem(dimension=10, lower_bound=-5.0, upper_bound=5.0)
```

### Rastrigin Function (Multimodal)
```python
problem = RastriginProblem(dimension=10, lower_bound=-5.12, upper_bound=5.12)
```

### Ackley Function (Multimodal)
```python
problem = AckleyProblem(dimension=10, lower_bound=-5.0, upper_bound=5.0)
```

### Griewank Function (Multimodal)
```python
problem = GriewankProblem(dimension=10, lower_bound=-600.0, upper_bound=600.0)
```

## Integration

The GBO implementation integrates with the project's core framework:
- Uses `ProblemInterface` for problem definitions
- Compatible with `Solution` class
- Supports visualization and logging
- Follows consistent API patterns

## References

- Ahmadianfar, I., Bozorg-Haddad, O., & Chu, X. (2020). Gradient-based optimizer: A new metaheuristic optimization algorithm.
- Ahmadianfar, I., et al. (2021). Gradient-based optimizer for solving systems of nonlinear equations.
- Various survey papers on population-based metaheuristics and GBO variants.

## Algorithm Variants

### Standard GBO
Classical implementation with fixed parameters.

### Adaptive GBO
Parameters adjust based on iteration progress.

### Enhanced GBO
Multiple LEO strategies and improved exploration mechanisms.

### Discrete GBO
Adapted for permutation-based problems like TSP.