#!/bin/python
"""
Unified entry point for running TSP metaheuristics using different hybrid approaches.

This script allows users to select from various hybrid metaheuristic approaches 
for solving the Traveling Salesperson Problem (TSP).
"""
import argparse
from TSP.solvers.Hybrid.RoundRobin.round_robin import run_hybrid_round_robin
from TSP.solvers.Hybrid.Parallel.parallel import run_hybrid_parallel
from TSP.solvers.Hybrid.Phased.phased_solver import run_hybrid_phased

def main():
    """Parse command line arguments and run the selected hybrid approach."""
    parser = argparse.ArgumentParser(
        description="Run hybrid metaheuristic approaches for the TSP problem."
    )
    
    # Add arguments for the hybrid approach
    parser.add_argument(
        "--approach", 
        "-a", 
        default="round_robin",
        choices=["round_robin", "parallel", "phased"], 
        help="The hybrid approach to use (default: round_robin)"
    )
    
    # Add arguments for TSP parameters
    parser.add_argument(
        "--cities", 
        "-c", 
        type=int, 
        default=20,
        help="Number of cities in the TSP problem (default: 20)"
    )
    parser.add_argument(
        "--population", 
        "-p", 
        type=int, 
        default=500,
        help="Population size for each algorithm (default: 500)"
    )
    parser.add_argument(
        "--iterations", 
        "-i", 
        type=int, 
        default=2000,
        help="Maximum number of iterations (default: 2000)"
    )
    parser.add_argument(
        "--seed", 
        "-s", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--sharing-interval",
        type=int,
        default=10,
        help="How often to share solutions between algorithms in parallel mode (default: 10)"
    )
    parser.add_argument(
        "--no-visualize", 
        action="store_true",
        help="Disable visualization of results"
    )
    parser.add_argument(
        "--no-save-plots", 
        action="store_true",
        help="Disable saving plot images"
    )
    
    # Add phased-specific arguments
    parser.add_argument(
        "--igwo-share",
        type=float,
        default=0.33,
        help="Proportion of iterations for IGWO in phased approach (default: 0.33)"
    )
    parser.add_argument(
        "--gwo-share",
        type=float,
        default=0.33,
        help="Proportion of iterations for GWO in phased approach (default: 0.33)"
    )
    parser.add_argument(
        "--ga-mutation",
        type=float,
        default=0.1,
        help="Mutation rate for GA in phased approach (default: 0.1)"
    )
    parser.add_argument(
        "--ga-crossover",
        type=float,
        default=0.8,
        help="Crossover rate for GA in phased approach (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    # Common parameters for all approaches
    common_params = {
        "num_cities": args.cities,
        "population_size": args.population,
        "max_iterations": args.iterations,
        "seed": args.seed,
        "visualize": not args.no_visualize,
        "save_route_plot": not args.no_save_plots,
        "save_convergence_plot": not args.no_save_plots
    }
    
    # Run the selected hybrid approach
    if args.approach == "round_robin":
        run_hybrid_round_robin(**common_params)
    elif args.approach == "parallel":
        run_hybrid_parallel(sharing_interval=args.sharing_interval, **common_params)
    elif args.approach == "phased":
        phased_params = {
            "igwo_share": args.igwo_share,
            "gwo_share": args.gwo_share,
            "ga_mutation_rate": args.ga_mutation,
            "ga_crossover_rate": args.ga_crossover
        }
        run_hybrid_phased(**common_params, **phased_params)
    else:
        print(f"Hybrid approach '{args.approach}' is not implemented yet.")

if __name__ == "__main__":
    main()
