# TDG: Test Data Generation

A modular framework for meta-heuristic optimization algorithms directed by Reinforcement Learning Agent, with a focus on knowledge transfer in inheritance hierarchy.

## Project Structure

- `Core/`: Core interfaces and framework components
  - `problem.py`: Problem interface and Solution class
  - `search_algorithm.py`: SearchAlgorithm abstract base class
  - `orchestrator.py`: HybridSearchOrchestrator for combining approaches

- `TSP/`: Traveling Salesperson Problem implementation
  - `TSP.py`: TSP problem definition
  - `solvers/`: Algorithm implementations for TSP
    - `GA/`: Genetic Algorithm implementation
    - `GWO/`: Gray Wolf Optimization implementation
    - `IGWO/`: Improved Gray Wolf Optimization implementation
    - `Hybrid/`: Hybrid metaheuristic approaches
      - `RoundRobin/`: Round-robin approach (alternating between algorithms)
      - `Parallel/`: Parallel approach (running algorithms simultaneously)

- `GA/`: Generic Genetic Algorithm implementation
- `GWO/`: Generic Gray Wolf Optimization implementation
- `IGWO/`: Generic Improved Gray Wolf Optimization implementation

## Hybrid Metaheuristic Approaches

The framework supports multiple hybrid approaches for combining metaheuristic algorithms:

1. **Round Robin**: Alternates between different algorithms in a cyclic manner
2. **Parallel**: Runs all algorithms simultaneously and shares solutions periodically

## Usage

Run the main script with the desired parameters:

```bash
python main.py --approach round_robin --cities 20 --iterations 2000
```

### Available Arguments:

- `--approach` or `-a`: Hybrid approach to use (`round_robin` or `parallel`)
- `--cities` or `-c`: Number of cities in the TSP problem
- `--population` or `-p`: Population size for each algorithm
- `--iterations` or `-i`: Maximum number of iterations
- `--seed` or `-s`: Random seed for reproducibility
- `--sharing-interval`: How often to share solutions between algorithms in parallel mode
- `--no-visualize`: Disable visualization of results
- `--no-save-plots`: Disable saving plot images

## References

See `REFACTORING_PLAN.md` for detailed refactoring steps and design decisions.
