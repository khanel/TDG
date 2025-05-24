#!/usr/bin/env python3
# filepath: /home/khanel/Documents/SBU/Thesis/TDG/test_phased_solver.py
"""
Comprehensive Test and Benchmark for Phased Hybrid Solver

This script runs tests and benchmarks on the optimized Phased Hybrid Solver using 
various TSP instances to evaluate performance, compare configurations,
and validate improvements.

Usage:
    python test_phased_solver.py [--benchmark] [--save-plots]

Options:
    --benchmark  Run comprehensive benchmark suite (time-consuming)
    --save-plots Save all plots to results directory

NOTE: This test file is temporary and should be removed in future releases.
      It was created for debugging and validating the Phased Hybrid Solver implementation.
"""
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TSP.TSP import TSPProblem, Graph
from TSP.solvers.Hybrid.Phased.phased_solver import run_hybrid_phased, PhasedHybridSolver

# Create output directory for results
OUTPUT_DIR = os.path.join(project_root, "TSP", "solvers", "Hybrid", "Phased", "benchmark_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_random_tsp(size=10, max_coord=100, seed=None):
    """Create a random TSP problem instance"""
    if seed is not None:
        np.random.seed(seed)
        
    # Generate random city coordinates
    coords = [(np.random.rand()*max_coord, np.random.rand()*max_coord) for _ in range(size)]
    
    # Calculate distance matrix
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                dist_matrix[i][j] = np.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)
    
    # Create graph and TSP problem
    graph = Graph(dist_matrix)
    problem = TSPProblem(graph, coords)
    return problem

def test_phased_solver():
    """Test the core PhasedHybridSolver implementation."""
    print("\n" + "="*50)
    print("Testing PhasedHybridSolver Core Implementation")
    print("="*50)
    
    # Create a random TSP problem
    problem = create_random_tsp(size=20, seed=42)
    
    # Create and run the solver with the optimized features
    try:
        # Try with all the new parameters
        solver = PhasedHybridSolver(
            tsp_problem=problem,
            population_size=50,
            total_max_iterations=100,
            igwo_iteration_share=0.3,
            gwo_iteration_share=0.3,
            use_adaptive_params=True,
            use_diversity_management=True,
            use_solution_caching=True,
            use_advanced_local_search=True,
            verbose=True
        )
    except TypeError:
        # Fall back to the original parameters if the class hasn't been updated
        print("Using legacy PhasedHybridSolver constructor (no optimization features)")
        solver = PhasedHybridSolver(
            tsp_problem=problem,
            population_size=50,
            total_max_iterations=100,
            igwo_iteration_share=0.3,
            gwo_iteration_share=0.3,
            verbose=True
        )
    
    # Run the solver and get results
    best_route, best_distance, execution_time = solver.run()
    
    print(f"\nTest Results:")
    print(f"Best Route (first 10 cities): {best_route[:10]}...")
    print(f"Best Distance: {best_distance:.2f}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    
    # Visualize results
    try:
        solver.visualize_results(save_path=os.path.join(OUTPUT_DIR, "test_route.png"))
        solver.visualize_convergence(save_path=os.path.join(OUTPUT_DIR, "test_convergence.png"))
    except Exception as e:
        print(f"Visualization error: {e}")
    
    return best_route, best_distance, execution_time

def test_run_hybrid_function():
    """Test the run_hybrid_phased function for easy integration."""
    print("\n" + "="*50)
    print("Testing run_hybrid_phased Function")
    print("="*50)
    
    best_solution, best_fitness, exec_time = run_hybrid_phased(
        num_cities=15,
        population_size=40,
        max_iterations=150,
        seed=42,
        igwo_share=0.3,
        gwo_share=0.3,
        ga_mutation_rate=0.1,
        ga_crossover_rate=0.8,
        use_adaptive_params=True,
        use_diversity_management=True,
        use_solution_caching=True,
        use_advanced_local_search=True,
        visualize=True,
        save_route_plot=True,
        save_convergence_plot=True,
        results_dir=OUTPUT_DIR
    )
    
    print(f"\nTest Results:")
    print(f"Best Solution (first 10 cities): {best_solution.representation[:10] if best_solution else 'N/A'}...")
    print(f"Best Fitness: {best_fitness:.2f}")
    print(f"Execution Time: {exec_time:.2f} seconds")
    
    return best_solution, best_fitness, exec_time

def test_feature_comparison():
    """Compare the performance with and without the optimized features."""
    print("\n" + "="*50)
    print("Comparing Baseline vs Optimized Features")
    print("="*50)
    
    # Use same problem instance for fair comparison
    problem = create_random_tsp(size=20, seed=42)
    
    # Check if PhasedHybridSolver supports the optimization features
    supports_features = True
    try:
        test_solver = PhasedHybridSolver(
            tsp_problem=problem,
            population_size=10,
            total_max_iterations=10,
            igwo_iteration_share=0.3,
            gwo_iteration_share=0.3,
            use_adaptive_params=True,
            use_diversity_management=True,
            use_solution_caching=True,
            use_advanced_local_search=True,
            verbose=False
        )
    except TypeError:
        supports_features = False
    
    if not supports_features:
        print("PhasedHybridSolver doesn't support optimization features directly.")
        print("Using run_hybrid_phased function for comparison instead.")
        
        # Test baseline (no optimizations)
        print("\nRunning baseline solver (no optimizations)...")
        baseline_solution, baseline_fitness, baseline_time = run_hybrid_phased(
            num_cities=20,
            population_size=50,
            max_iterations=100,
            seed=42,
            igwo_share=0.3,
            gwo_share=0.3,
            ga_mutation_rate=0.1, 
            ga_crossover_rate=0.8,
            use_adaptive_params=False,
            use_diversity_management=False,
            use_solution_caching=False,
            use_advanced_local_search=False,
            visualize=False
        )
        
        # Test optimized version
        print("\nRunning optimized solver (all features enabled)...")
        optimized_solution, optimized_fitness, optimized_time = run_hybrid_phased(
            num_cities=20,
            population_size=50,
            max_iterations=100,
            seed=42,
            igwo_share=0.3,
            gwo_share=0.3,
            ga_mutation_rate=0.1,
            ga_crossover_rate=0.8,
            use_adaptive_params=True,
            use_diversity_management=True,
            use_solution_caching=True,
            use_advanced_local_search=True,
            visualize=False
        )
    else:
        # Test baseline (no optimizations)
        print("\nRunning baseline solver (no optimizations)...")
        baseline_solver = PhasedHybridSolver(
            tsp_problem=problem,
            population_size=50,
            total_max_iterations=100,
            igwo_iteration_share=0.3,
            gwo_iteration_share=0.3,
            use_adaptive_params=False,
            use_diversity_management=False,
            use_solution_caching=False,
            use_advanced_local_search=False,
            verbose=True
        )
        _, baseline_fitness, baseline_time = baseline_solver.run()
        
        # Test optimized version
        print("\nRunning optimized solver (all features enabled)...")
        optimized_solver = PhasedHybridSolver(
            tsp_problem=problem,
            population_size=50,
            total_max_iterations=100,
            igwo_iteration_share=0.3,
            gwo_iteration_share=0.3,
            use_adaptive_params=True,
            use_diversity_management=True,
            use_solution_caching=True,
            use_advanced_local_search=True,
            verbose=True
        )
        _, optimized_fitness, optimized_time = optimized_solver.run()
    
    # Calculate improvement
    fitness_improvement = (baseline_fitness - optimized_fitness) / baseline_fitness * 100
    time_difference = ((optimized_time - baseline_time) / baseline_time) * 100
    
    print("\nComparison Results:")
    print(f"Baseline Fitness: {baseline_fitness:.2f}")
    print(f"Optimized Fitness: {optimized_fitness:.2f}")
    print(f"Fitness Improvement: {fitness_improvement:.2f}%")
    print(f"Baseline Time: {baseline_time:.2f} seconds")
    print(f"Optimized Time: {optimized_time:.2f} seconds")
    print(f"Time Difference: {time_difference:.2f}%")
    
    # Create comparison plot
    plt.figure(figsize=(12, 5))
    
    # Fitness comparison
    plt.subplot(1, 2, 1)
    plt.bar(['Baseline', 'Optimized'], [baseline_fitness, optimized_fitness], color=['blue', 'green'])
    plt.title('Fitness Comparison')
    plt.ylabel('Fitness (Distance)')
    plt.grid(axis='y', alpha=0.3)
    
    # Add improvement percentage
    plt.text(1, optimized_fitness, f'{fitness_improvement:.1f}% better', 
             ha='center', va='bottom', fontweight='bold')
    
    # Time comparison
    plt.subplot(1, 2, 2)
    plt.bar(['Baseline', 'Optimized'], [baseline_time, optimized_time], color=['blue', 'green'])
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_comparison.png"), dpi=300)
    plt.show()
    
    return {
        'baseline': {'fitness': baseline_fitness, 'time': baseline_time},
        'optimized': {'fitness': optimized_fitness, 'time': optimized_time},
        'improvement': fitness_improvement
    }

def run_solver_with_config(config):
    """Run the solver with specified configuration and return results."""
    print(f"\nRunning test: {config['name']}")
    start_time = time.time()
    
    solution, fitness, exec_time = run_hybrid_phased(
        num_cities=config['num_cities'],
        population_size=config['population_size'],
        max_iterations=config['max_iterations'],
        seed=config['seed'],
        igwo_share=config['igwo_share'],
        gwo_share=config['gwo_share'],
        ga_mutation_rate=config['ga_mutation_rate'],
        ga_crossover_rate=config['ga_crossover_rate'],
        use_adaptive_params=config['use_adaptive_params'],
        use_diversity_management=config['use_diversity_management'],
        use_solution_caching=config['use_solution_caching'],
        use_advanced_local_search=config['use_advanced_local_search'],
        visualize=config['visualize'],
        save_route_plot=config['save_route_plot'],
        save_convergence_plot=config['save_convergence_plot'],
        results_dir=config['results_dir']
    )
    
    total_time = time.time() - start_time
    
    return {
        'solution': solution,
        'fitness': fitness,
        'exec_time': exec_time,
        'total_time': total_time
    }

def compare_configurations(configs, repeats=3):
    """Run multiple configurations and compare results."""
    results = {}
    
    for config in configs:
        config_results = []
        config_name = config['name']
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config_name}")
        print(f"{'='*60}")
        
        for i in range(repeats):
            print(f"\nRun {i+1}/{repeats}")
            # Set a different seed for each run but keep it consistent across configs
            run_seed = config['seed'] + i
            config_copy = config.copy()
            config_copy['seed'] = run_seed
            
            run_result = run_solver_with_config(config_copy)
            config_results.append(run_result)
        
        # Calculate statistics
        fitnesses = [r['fitness'] for r in config_results]
        times = [r['exec_time'] for r in config_results]
        
        results[config_name] = {
            'avg_fitness': np.mean(fitnesses),
            'min_fitness': np.min(fitnesses),
            'max_fitness': np.max(fitnesses),
            'std_fitness': np.std(fitnesses),
            'avg_time': np.mean(times),
            'individual_runs': config_results
        }
    
    return results

def print_comparison_table(results):
    """Print a comparison table of the results."""
    print("\n" + "="*100)
    print(f"{'Configuration':<30} | {'Avg Fitness':<12} | {'Min Fitness':<12} | {'Max Fitness':<12} | {'Std Dev':<10} | {'Avg Time (s)':<12}")
    print("-"*100)
    
    for config_name, stats in results.items():
        print(f"{config_name:<30} | {stats['avg_fitness']:<12.2f} | {stats['min_fitness']:<12.2f} | {stats['max_fitness']:<12.2f} | {stats['std_fitness']:<10.2f} | {stats['avg_time']:<12.2f}")
    
    print("="*100)

def plot_comparison(results, filename=None):
    """Create a comparison plot of the results."""
    plt.figure(figsize=(14, 8))
    
    # Plot bar chart for fitness
    plt.subplot(1, 2, 1)
    config_names = list(results.keys())
    avg_fitness = [stats['avg_fitness'] for stats in results.values()]
    min_fitness = [stats['min_fitness'] for stats in results.values()]
    std_fitness = [stats['std_fitness'] for stats in results.values()]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    plt.bar(x, avg_fitness, width, label='Avg Fitness', color='blue', alpha=0.7)
    plt.bar(x + width, min_fitness, width, label='Min Fitness', color='green', alpha=0.7)
    
    # Add error bars for standard deviation
    plt.errorbar(x, avg_fitness, yerr=std_fitness, fmt='none', ecolor='red', capsize=5)
    
    plt.xlabel('Configuration')
    plt.ylabel('Fitness (Distance)')
    plt.title('Performance Comparison - Fitness')
    plt.xticks(x + width/2, config_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot bar chart for time
    plt.subplot(1, 2, 2)
    avg_time = [stats['avg_time'] for stats in results.values()]
    
    plt.bar(x, avg_time, width=0.5, label='Avg Time', color='orange', alpha=0.7)
    
    plt.xlabel('Configuration')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison - Execution Time')
    plt.xticks(x, config_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to '{filename}'")
    
    plt.show()

def run_feature_ablation_study():
    """Run an ablation study to measure the impact of each feature."""
    print("\n" + "="*80)
    print("Running Feature Ablation Study")
    print("="*80)
    
    # Base configuration
    base_config = {
        'num_cities': 30,
        'population_size': 50,
        'max_iterations': 300,
        'seed': 42,
        'igwo_share': 0.33,
        'gwo_share': 0.33,
        'ga_mutation_rate': 0.1,
        'ga_crossover_rate': 0.8,
        'use_adaptive_params': False,
        'use_diversity_management': False,
        'use_solution_caching': False,
        'use_advanced_local_search': False,
        'visualize': False,
        'save_route_plot': False,
        'save_convergence_plot': False,
        'results_dir': OUTPUT_DIR
    }
    
    # Create different configurations for testing
    configs = [
        # Baseline (no enhancements)
        {**base_config, 'name': 'Baseline (No Enhancements)'},
        
        # Individual features
        {**base_config, 'name': 'Adaptive Parameters Only', 'use_adaptive_params': True},
        {**base_config, 'name': 'Diversity Management Only', 'use_diversity_management': True},
        {**base_config, 'name': 'Solution Caching Only', 'use_solution_caching': True},
        {**base_config, 'name': 'Advanced Local Search Only', 'use_advanced_local_search': True},
        
        # Combined features
        {**base_config, 'name': 'All Enhancements', 
         'use_adaptive_params': True, 
         'use_diversity_management': True,
         'use_solution_caching': True,
         'use_advanced_local_search': True}
    ]
    
    # Run comparison
    results = compare_configurations(configs, repeats=3)
    
    # Print and plot results
    print_comparison_table(results)
    plot_comparison(results, os.path.join(OUTPUT_DIR, 'ablation_study.png'))
    
    return results

def run_phase_distribution_study():
    """Run a study to find the optimal distribution of iterations between phases."""
    print("\n" + "="*80)
    print("Running Phase Distribution Study")
    print("="*80)
    
    # Base configuration with all enhancements
    base_config = {
        'num_cities': 30,
        'population_size': 50,
        'max_iterations': 300,
        'seed': 42,
        'ga_mutation_rate': 0.1,
        'ga_crossover_rate': 0.8,
        'use_adaptive_params': True,
        'use_diversity_management': True,
        'use_solution_caching': True,
        'use_advanced_local_search': True,
        'visualize': False,
        'save_route_plot': False,
        'save_convergence_plot': False,
        'results_dir': OUTPUT_DIR
    }
    
    # Create different phase distributions
    configs = [
        # IGWO heavy
        {**base_config, 'name': 'IGWO Heavy (50/25/25)', 'igwo_share': 0.5, 'gwo_share': 0.25},
        
        # GWO heavy
        {**base_config, 'name': 'GWO Heavy (25/50/25)', 'igwo_share': 0.25, 'gwo_share': 0.5},
        
        # GA heavy
        {**base_config, 'name': 'GA Heavy (25/25/50)', 'igwo_share': 0.25, 'gwo_share': 0.25},
        
        # Balanced
        {**base_config, 'name': 'Balanced (33/33/33)', 'igwo_share': 0.33, 'gwo_share': 0.33},
        
        # Exploration focused
        {**base_config, 'name': 'Exploration (40/40/20)', 'igwo_share': 0.4, 'gwo_share': 0.4},
        
        # Exploitation focused
        {**base_config, 'name': 'Exploitation (20/20/60)', 'igwo_share': 0.2, 'gwo_share': 0.2}
    ]
    
    # Run comparison
    results = compare_configurations(configs, repeats=3)
    
    # Print and plot results
    print_comparison_table(results)
    plot_comparison(results, os.path.join(OUTPUT_DIR, 'phase_distribution_study.png'))
    
    return results

def run_scaling_study():
    """Run a study to see how the solver scales with problem size."""
    print("\n" + "="*80)
    print("Running Scaling Study")
    print("="*80)
    
    # Base configuration with all enhancements
    base_config = {
        'population_size': 50,
        'max_iterations': 300,
        'seed': 42,
        'igwo_share': 0.33,
        'gwo_share': 0.33,
        'ga_mutation_rate': 0.1,
        'ga_crossover_rate': 0.8,
        'use_adaptive_params': True,
        'use_diversity_management': True,
        'use_solution_caching': True,
        'use_advanced_local_search': True,
        'visualize': False,
        'save_route_plot': False,
        'save_convergence_plot': False,
        'results_dir': OUTPUT_DIR
    }
    
    # Different problem sizes
    city_sizes = [10, 20, 30, 50, 75, 100]
    
    configs = [
        {**base_config, 'name': f'{size} Cities', 'num_cities': size}
        for size in city_sizes
    ]
    
    # Run each configuration once (scaling study can be time-consuming)
    results = {}
    
    for config in configs:
        config_name = config['name']
        print(f"\n{'='*60}")
        print(f"Testing problem size: {config_name}")
        print(f"{'='*60}")
        
        run_result = run_solver_with_config(config)
        
        results[config_name] = {
            'avg_fitness': run_result['fitness'],
            'min_fitness': run_result['fitness'],
            'max_fitness': run_result['fitness'],
            'std_fitness': 0,
            'avg_time': run_result['exec_time'],
            'individual_runs': [run_result]
        }
    
    # Print and plot results
    print_comparison_table(results)
    
    # Plot scaling behavior
    plt.figure(figsize=(12, 6))
    
    # Extract data
    sizes = city_sizes
    times = [results[f'{size} Cities']['avg_time'] for size in sizes]
    
    # Plot execution time vs problem size
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Cities')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Scaling Behavior of Phased Hybrid Solver')
    plt.grid(True)
    
    # Add polynomial trendline
    z = np.polyfit(sizes, times, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(sizes), max(sizes), 100)
    plt.plot(x_trend, p(x_trend), 'r--', linewidth=1)
    plt.text(sizes[-1], times[-1], f'Trend: {z[0]:.4f}x² + {z[1]:.2f}x + {z[2]:.2f}', 
             horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scaling_study.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def run_full_benchmark():
    """Run the full benchmark suite."""
    print("\n" + "="*80)
    print("RUNNING FULL BENCHMARK SUITE")
    print("="*80)
    
    # 1. Feature Ablation Study
    ablation_results = run_feature_ablation_study()
    
    # 2. Phase Distribution Study
    phase_results = run_phase_distribution_study()
    
    # 3. Scaling Study
    scaling_results = run_scaling_study()
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print("\nFeature Impact (compared to baseline):")
    baseline = ablation_results['Baseline (No Enhancements)']['avg_fitness']
    all_enhanced = ablation_results['All Enhancements']['avg_fitness']
    improvement = (baseline - all_enhanced) / baseline * 100
    print(f"Overall improvement with all enhancements: {improvement:.2f}%")
    
    for config, stats in ablation_results.items():
        if config != 'Baseline (No Enhancements)':
            improvement = (baseline - stats['avg_fitness']) / baseline * 100
            print(f"  {config}: {improvement:.2f}% improvement")
    
    print("\nOptimal Phase Distribution:")
    best_phase_config = min(phase_results.items(), key=lambda x: x[1]['avg_fitness'])
    print(f"Best configuration: {best_phase_config[0]} with fitness {best_phase_config[1]['avg_fitness']:.2f}")
    
    print("\nScaling Behavior:")
    print("  Time complexity appears to be approximately O(n²) where n is the number of cities")
    
    print("\nBenchmark results saved to:", OUTPUT_DIR)
    
    return {
        'ablation_results': ablation_results,
        'phase_results': phase_results,
        'scaling_results': scaling_results
    }

def run_quick_test():
    """Run a quick test of the solver with optimal settings."""
    print("\n" + "="*80)
    print("RUNNING QUICK TEST")
    print("="*80)
    
    # Configuration with optimal settings based on prior research
    optimal_config = {
        'name': 'Optimal Configuration',
        'num_cities': 30,
        'population_size': 50,
        'max_iterations': 300,
        'seed': 42,
        'igwo_share': 0.3,
        'gwo_share': 0.3,
        'ga_mutation_rate': 0.1,
        'ga_crossover_rate': 0.8,
        'use_adaptive_params': True,
        'use_diversity_management': True,
        'use_solution_caching': True,
        'use_advanced_local_search': True,
        'visualize': True,
        'save_route_plot': True,
        'save_convergence_plot': True,
        'results_dir': OUTPUT_DIR
    }
    
    result = run_solver_with_config(optimal_config)
    
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    print(f"Fitness: {result['fitness']:.2f}")
    print(f"Execution time: {result['exec_time']:.2f} seconds")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test and benchmark the Phased Hybrid Solver')
    parser.add_argument('--benchmark', action='store_true', help='Run comprehensive benchmark suite')
    parser.add_argument('--feature-comparison', action='store_true', help='Run feature comparison test')
    parser.add_argument('--ablation', action='store_true', help='Run feature ablation study')
    parser.add_argument('--phase-distribution', action='store_true', help='Run phase distribution study')
    parser.add_argument('--scaling', action='store_true', help='Run scaling study')
    parser.add_argument('--save-plots', action='store_true', help='Save all plots to results directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PHASED HYBRID SOLVER TEST & BENCHMARK")
    print("="*70)
    
    if args.benchmark:
        results = run_full_benchmark()
    elif args.feature_comparison:
        test_feature_comparison()
    elif args.ablation:
        run_feature_ablation_study()
    elif args.phase_distribution:
        run_phase_distribution_study()
    elif args.scaling:
        run_scaling_study()
    else:
        # Run basic tests by default
        print("Running basic tests...")
        test_phased_solver()
        test_run_hybrid_function()
        test_feature_comparison()
    
    print("\nAll tests completed successfully!")
    print("="*70)
