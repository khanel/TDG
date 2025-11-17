# Candidate Design Pattern: Memetic Algorithm (MA)

## 1. High-Level Description

A **Memetic Algorithm (MA)** is not a single, specific algorithm, but rather a powerful **hybrid design pattern** that enhances a population-based global search with individual local refinement. It is inspired by the concept of a "meme" as a unit of cultural evolution, analogous to a gene in biological evolution.

The core philosophy of an MA is to combine the strengths of two different types of algorithms:

1.  **Global Search (Evolution)**: A population-based algorithm like a Genetic Algorithm (GA) or Particle Swarm Optimization (PSO) is used to broadly explore the search space and identify promising regions. This is the "evolutionary" component.
2.  **Local Search (Individual Learning)**: A local search heuristic, such as Hill Climbing or Simulated Annealing, is applied to some or all individuals in the population after the global search step. This process refines the solutions, pulling them towards the nearest local optimum. This is the "individual learning" or "meme" component.

By combining these two strategies, MAs create a powerful synergy that balances exploration (finding good hills) and exploitation (climbing the hills to their peak). This pattern is perfectly suited for your framework, as it formalizes the concept of using multiple algorithms cooperatively.

## 2. Pseudocode (Conceptual)

This pseudocode is conceptual, showing how an MA would be structured within your framework by composing two `SearchAlgorithm` instances.

```
// --- Initialization ---
// GlobalSearcher could be GA, PSO, GWO, etc.
// LocalSearcher could be HillClimbing, RandomWalk, etc.
Initialize GlobalSearcher algorithm
Initialize LocalSearcher algorithm
Initialize population P using GlobalSearcher.initialize()

// --- Main Loop ---
for generation g = 1 to max_generations do
    
    // 1. Global Search Step
    // The GlobalSearcher performs its standard step (e.g., selection, crossover, mutation)
    // to produce a new population of candidate solutions.
    P_intermediate = GlobalSearcher.step(P)
    
    // 2. Local Refinement Step (Individual Learning)
    P_refined = empty population
    for each individual 'sol' in P_intermediate do
        // For each individual, run the LocalSearcher for a few steps
        // to find a nearby local optimum.
        refined_sol = LocalSearcher.refine(sol, local_search_depth)
        add refined_sol to P_refined
    end for
    
    // 3. Update the main population
    P = P_refined
    
    // Update best solution found so far
    
end for

Return the best individual found in P
```
*Note: `LocalSearcher.refine` would be a special method that takes a single solution, runs its `step` logic for `local_search_depth` iterations starting from that solution, and returns the best solution it found.*

## 3. Reference

**Title**: "Memetic algorithms: A state-of-the-art review"
**Authors**: P. Moscato, C. Cotta
**URL**: https://www.sciencedirect.com/science/article/abs/pii/S0377221702007327 
(This is a classic and highly-cited review paper on the topic).
