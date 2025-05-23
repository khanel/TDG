# Round Robin Hybrid Approach

The Round Robin approach alternates between different metaheuristic algorithms in a cyclic manner. In each iteration, a different algorithm is selected to evolve the population. This approach leverages the strengths of each algorithm at different stages of the search process.

## Implementation Details

The implementation in `round_robin.py` provides:

1. A `run_hybrid_round_robin` function that accepts various parameters to configure the TSP problem and the hybrid approach
2. Initialization of multiple algorithms (GA, GWO, IGWO) with a common problem instance
3. A main loop that cycles through the algorithms, using one per iteration
4. Tracking of the best solution found across all algorithms
5. Visualization of the final route and convergence history

## Visualization

All visualizations are saved in this directory:
- `route.png`: The best route found by the hybrid approach
- `convergence.png`: The convergence history of the best solution fitness over iterations

## Usage

This approach can be used through the main script:

```bash
python main.py --approach round_robin --cities 20 --iterations 2000
```

## Parameters

- `num_cities`: Number of cities in the TSP problem
- `population_size`: Size of the population for each algorithm
- `max_iterations`: Maximum number of iterations
- `seed`: Random seed for reproducibility
- `visualize`: Whether to visualize the results
- `save_route_plot`: Whether to save the route plot
- `save_convergence_plot`: Whether to save the convergence plot
- `results_dir`: Directory to save results (defaults to RoundRobin/)
