# Particle Swarm Optimization (PSO) Implementation

This directory contains the implementation of Particle Swarm Optimization (PSO) as described in `docs/PSO.md`.

## Files

- `PSO.py`: Main Particle Swarm Optimization implementation
- `Problem.py`: Problem definitions and utilities for PSO
- `README.md`: This documentation file

## Overview

Particle Swarm Optimization simulates a swarm of particles that fly through the search space, influenced by their own best positions and the swarm's global best position. This implementation supports both the classical inertia weight formulation and the constriction factor formulation for guaranteed convergence.

## Key Features

### Algorithm Variants
- **Inertia Weight PSO**: Classical formulation with ω parameter
- **Constriction Factor PSO**: Guaranteed convergence with χ parameter
- **Adaptive Exploration**: Parameters adjust based on iteration progress
- **Enhanced Exploration**: Multiple strategies for improved search

### PSO Components
- **Particles**: Candidate solutions with position and velocity
- **Personal Best**: Each particle's best position found so far
- **Global Best**: Swarm's best position across all particles
- **Velocity Update**: Combination of inertia, cognitive, and social components

## Usage

### Basic Usage

```python
from algorithms.PSO.PSO import ParticleSwarmOptimization
from algorithms.PSO.Problem import ContinuousOptimizationProblem, create_pso_parameters

# Create problem
problem = ContinuousOptimizationProblem(dimension=10, lower_bound=-5.0, upper_bound=5.0)

# Create PSO parameters
params = create_pso_parameters(population_size=30, problem_type='continuous')

# Create and run PSO
pso = ParticleSwarmOptimization(
    problem=problem,
    population_size=30,
    max_iterations=100,
    verbosity=1,
    **params
)

# Initialize and run
pso.initialize()
for _ in range(100):
    pso.step()

# Get best solution
best_solution = pso.get_best_solution()
print(f"Best fitness: {best_solution.fitness}")
```

### Constriction Factor Usage

```python
# Create PSO with constriction factor for guaranteed convergence
pso = ParticleSwarmOptimization(
    problem=problem,
    population_size=30,
    c1=2.05,  # c1 + c2 > 4 for constriction
    c2=2.05,
    use_constriction=True,  # Automatically calculates chi
    max_iterations=100,
    verbosity=1
)
```

### Enhanced Exploration Usage

```python
# Create PSO with enhanced exploration
pso = ParticleSwarmOptimization(
    problem=problem,
    population_size=30,
    max_iterations=100,
    verbosity=1,
    exploration_boost=1.5,      # 1.5x exploration enhancement
    adaptive_exploration=True,  # Adaptive exploration over time
    **params
)

# Features:
# - Early iterations: High exploration (boosted parameters)
# - Mid iterations: Balanced exploration/exploitation
# - Late iterations: Focus on exploitation
# - Adaptive parameter adjustment
```

## Parameters

### Core Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `population_size` | Number of particles | 30 | 10-100 |
| `omega` | Inertia weight | 0.7 | 0.4-0.9 |
| `c1` | Cognitive coefficient | 1.5 | 1.0-2.5 |
| `c2` | Social coefficient | 1.5 | 1.0-2.5 |
| `max_iterations` | Maximum iterations | 100 | 50-500 |
| `vmax_factor` | Max velocity factor | 0.2 | 0.1-0.5 |

### Constriction Parameters
| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `use_constriction` | Use constriction factor | False | Set to True for guaranteed convergence |
| `c1 + c2` | Sum must be > 4 | 4.1 | Typical: c1=c2=2.05 |

### Enhanced Exploration Parameters
| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `exploration_boost` | Exploration multiplier | 1.0 | > 1.0 increases exploration |
| `adaptive_exploration` | Adaptive parameters | False | Enables time-based adaptation |

## Algorithm Variants

### 1. Inertia Weight PSO
```python
v[t+1] = ω × v[t] + c1 × r1 × (p - x[t]) + c2 × r2 × (g - x[t])
x[t+1] = x[t] + v[t+1]
```

### 2. Constriction Factor PSO
```python
v[t+1] = χ × (v[t] + c1 × r1 × (p - x[t]) + c2 × r2 × (g - x[t]))
x[t+1] = x[t] + v[t+1]
```
Where χ = 2 / (|c1 + c2| - 2 + √((c1 + c2)² - 4(c1 + c2)))

## Problem Types

### Continuous Problems
- Uses standard velocity updates
- Boundary handling with clipping
- Velocity clamping to prevent explosion

### Discrete Problems (TSP)
- Adapted for permutation-based optimization
- Swap-based position updates
- Maintains tour validity

## Velocity Handling

### Boundary Constraints
- **Clipping**: x = max(min(x, ub), lb)
- **Reflection**: Bounce off boundaries
- **Random**: Reinitialize out-of-bounds particles

### Velocity Clamping
- Prevents particles from moving too fast
- vmax = vmax_factor × (upper_bound - lower_bound)
- v = max(min(v, vmax), -vmax)

## Convergence Analysis

### Stability Conditions
- **Inertia Weight**: ω ∈ [0.4, 0.9] for stable convergence
- **Constriction Factor**: c1 + c2 > 4 guarantees convergence
- **Exploration Balance**: c1 and c2 control personal vs social influence

### Performance Metrics
- **Exploration Rate**: Measures particle diversity
- **Convergence Speed**: Iterations to reach target fitness
- **Solution Quality**: Best fitness achieved

## Advanced Features

### Adaptive Parameters
- **Time-based**: Parameters change with iteration progress
- **Performance-based**: Adjust based on convergence behavior
- **Multi-stage**: Different strategies for different phases

### Swarm Intelligence
- **Diversity Maintenance**: Prevents premature convergence
- **Information Sharing**: Global best guides entire swarm
- **Social Learning**: Particles learn from successful neighbors

## Example Problems

### Sphere Function (Continuous)
```python
problem = ContinuousOptimizationProblem(dimension=10, lower_bound=-5.0, upper_bound=5.0)
```

### Rastrigin Function (Multimodal)
```python
problem = RastriginProblem(dimension=10, lower_bound=-5.12, upper_bound=5.12)
```

### Rosenbrock Function (Valley)
```python
problem = RosenbrockProblem(dimension=2, lower_bound=-2.0, upper_bound=2.0)
```

## Integration

The PSO implementation integrates with the project's core framework:
- Uses `ProblemInterface` for problem definitions
- Compatible with `Solution` class
- Supports visualization and logging
- Follows consistent API patterns

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
- Clerc, M., & Kennedy, J. (2002). The particle swarm-explosion, stability, and convergence in a multidimensional complex space.
- Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.