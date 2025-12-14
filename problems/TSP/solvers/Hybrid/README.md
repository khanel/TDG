# Hybrid Metaheuristic Approaches for TSP

This directory contains various hybrid metaheuristic approaches for solving the Traveling Salesperson Problem (TSP). Each approach combines multiple search algorithms in different ways to achieve better performance than individual algorithms.

## Available Approaches

### Round Robin

**Directory**: `RoundRobin/`

The Round Robin approach alternates between different algorithms in a cyclic manner. In each iteration, a different algorithm is used to evolve the population. This helps leverage the strengths of each algorithm at different stages of the search.

**Implementation**: `RoundRobin/round_robin.py`

**Key Parameters**:
- `num_cities`: Number of cities in the TSP problem
- `population_size`: Size of the population for each algorithm
- `max_iterations`: Maximum number of iterations
- `seed`: Random seed for reproducibility
- `visualize`: Whether to visualize the results
- `save_route_plot`: Whether to save the route plot
- `save_convergence_plot`: Whether to save the convergence plot
- `results_dir`: Directory to save results (defaults to RoundRobin/)

### Parallel

**Directory**: `Parallel/`

The Parallel approach runs multiple algorithms simultaneously, with periodic sharing of the best solutions between them. This allows different algorithms to explore different regions of the search space and benefit from each other's discoveries.

**Implementation**: `Parallel/parallel.py`

**Key Parameters**:
- Same as Round Robin, plus:
- `sharing_interval`: How often to share solutions between algorithms (defaults to 10 iterations)

## Adding New Approaches

To add a new hybrid approach:

1. Create a new directory under `TSP/solvers/Hybrid/` for your approach
2. Implement your approach following the pattern of existing approaches
3. Update `main.py` to include your approach in the command-line options
4. Document your approach in this README.md file

## Usage

All hybrid approaches can be run through the main script:

```bash
python main.py --approach [approach_name] [options]
```

See the project's main README.md for a full list of available options.
