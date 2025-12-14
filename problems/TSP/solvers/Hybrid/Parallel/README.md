# Parallel Hybrid Approach

The Parallel approach runs multiple metaheuristic algorithms simultaneously, with periodic sharing of the best solutions between them. This allows different algorithms to explore different regions of the search space independently, while still benefiting from each other's discoveries.

## Implementation Details

The implementation in `parallel.py` provides:

1. A `run_hybrid_parallel` function that accepts various parameters to configure the TSP problem and the hybrid approach
2. Initialization of multiple algorithms (GA, GWO, IGWO) with a common problem instance
3. Parallel evolution of populations for each algorithm
4. Periodic sharing of the best solutions between algorithms at specified intervals
5. Tracking of the best solution found across all algorithms
6. Visualization of the final route and convergence history

## Solution Sharing

The key feature of this approach is the sharing of solutions between algorithms. At specified intervals (controlled by the `sharing_interval` parameter), each algorithm:
1. Contributes its best solution to a shared pool
2. Receives the best solution found by other algorithms
3. Incorporates the received solution into its population

This sharing mechanism enables algorithms to benefit from discoveries made by other algorithms and helps prevent premature convergence to local optima.

## Visualization

All visualizations are saved in this directory:
- `route.png`: The best route found by the hybrid approach
- `convergence.png`: The convergence history of the best solution fitness over iterations

## Usage

This approach can be used through the main script:

```bash
python main.py --approach parallel --cities 20 --iterations 2000 --sharing-interval 10
```

## Parameters

- `num_cities`: Number of cities in the TSP problem
- `population_size`: Size of the population for each algorithm
- `max_iterations`: Maximum number of iterations
- `seed`: Random seed for reproducibility
- `visualize`: Whether to visualize the results
- `save_route_plot`: Whether to save the route plot
- `save_convergence_plot`: Whether to save the convergence plot
- `sharing_interval`: How often to share solutions between algorithms (defaults to 10 iterations)
- `results_dir`: Directory to save results (defaults to Parallel/)
