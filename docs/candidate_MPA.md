# Candidate Algorithm: Marine Predators Algorithm (MPA)

## 1. High-Level Description

**Marine Predators Algorithm (MPA)** is a recent, nature-inspired optimization algorithm that mimics the foraging strategies of predators in the ocean. A key feature of MPA is its use of both **Lévy flight** and **Brownian motion** as models for predator movement, which provides a robust mix of exploration and exploitation behaviors.

The algorithm's strategy is explicitly divided into three main phases, dictated by the ratio of the current iteration to the maximum iterations, which simulates the changing velocity ratio between predator and prey throughout a chase:

*   **Phase 1 (High Velocity Ratio - Early Iterations)**: This is a pure exploration phase. The prey is moving faster than the predator, so the predator uses Lévy flight to perform broad, random searches across the solution space.
*   **Phase 2 (Unit Velocity Ratio - Middle Iterations)**: This is a transition phase where exploration and exploitation are performed simultaneously. The population is split: one half continues to explore (assigned to Lévy flight), while the other half begins to exploit (assigned to Brownian motion).
*   **Phase 3 (Low Velocity Ratio - Late Iterations)**: This is a pure exploitation phase. The predator is now faster than the prey and uses Brownian motion to perform a fine-grained, intensive local search around the most promising solutions.

MPA also considers environmental factors like **Fish Aggregating Devices (FADs)**, which are modeled as a mechanism to help the algorithm escape local optima.

## 2. Pseudocode

```
Initialize population of predators (solutions) X_i
Set max_iterations T

while (t < T) do
    // Evaluate fitness and find the best solution (Top_Predator)
    
    // --- Update Strategy based on Iteration ---
    
    // Phase 1: Exploration (for the first 1/3 of iterations)
    if t < T/3 then
        for each predator X_i do
            Step = Levy_dist() * (Top_Predator - Levy_dist() * X_i)
            X_i = X_i + 0.5 * rand() * Step
        end for
    end if
    
    // Phase 2: Transition (for the middle 1/3 of iterations)
    else if t < 2*T/3 then
        // First half of population exploits
        for i = 1 to N/2 do
            Step = Brownian_dist() * (Top_Predator - Brownian_dist() * X_i)
            X_i = X_i + 0.5 * rand() * Step
        end for
        // Second half of population explores
        for i = N/2 + 1 to N do
            Step = Levy_dist() * (Top_Predator - Levy_dist() * X_i)
            X_i = X_i + 0.5 * R * Step // R is a random vector
        end for
    end if
    
    // Phase 3: Exploitation (for the last 1/3 of iterations)
    else
        for each predator X_i do
            Step = Brownian_dist() * (Brownian_dist() * Top_Predator - X_i)
            X_i = Top_Predator + 0.5 * CF * Step // CF is an adaptive parameter
        end for
    end if
    
    // --- FADs Effect (Local Optima Escape) ---
    FADs_prob = 0.2
    for each predator X_i do
        if rand() < FADs_prob then
            U = binary vector (0 or 1)
            X_i = X_i + CF * ((LowerBound + rand()*(UpperBound-LowerBound)) * U)
        end if
    end for
    
    // Evaluate and update Top_Predator
    t = t + 1
end while

Return Top_Predator
```

## 3. Reference

**Title**: "Marine Predators Algorithm: A Nature-Inspired Metaheuristic"
**Authors**: A. Faramarzi, M. Heidarinejad, S. Mirjalili, A. Gandomi
**URL**: https://www.sciencedirect.com/science/article/pii/S095741742030025X
