# Candidate Algorithm: Whale Optimization Algorithm (WOA)

## 1. High-Level Description

The **Whale Optimization Algorithm (WOA)** is a popular swarm-based metaheuristic inspired by the hunting behavior of humpback whales. Specifically, it models the "bubble-net feeding" strategy, a unique foraging method where whales create a circle of bubbles to trap schools of krill or fish.

The algorithm's behavior is divided into two main phases:

1.  **Exploitation Phase (Bubble-net Attack)**: This phase mimics the attack on the prey. It has two coordinated mechanisms:
    *   **Shrinking Encircling**: The whales tighten their circle around the prey (the current best solution).
    *   **Spiral Updating**: The whales swim towards the prey in a logarithmic spiral path, simulating the bubble-net maneuver.
    The algorithm randomly chooses between these two movements to create a comprehensive local search.

2.  **Exploration Phase (Search for Prey)**: To ensure global search, the algorithm has a mechanism for whales to search more broadly. Instead of moving towards the best-known prey, a whale will update its position based on the location of another *randomly chosen* whale. This forces the population to explore new and potentially more promising areas of the search space.

The switch between exploration and exploitation is controlled by a coefficient that changes during the iterations, making the algorithm's search both focused and wide-ranging.

## 2. Pseudocode

```
Initialize the population of whales X_i
Set max_iterations T

while (t < T) do
    // Evaluate fitness for all whales and update X* (the best solution)
    
    // Update parameters a, A, C, l
    a = 2 - t * (2/T) // Linearly decreases from 2 to 0
    
    for each whale X_i do
        p = random number in [0, 1]
        A = 2 * a * rand() - a
        C = 2 * rand()
        
        if p < 0.5 then
            // --- Exploration or Shrinking Encircling ---
            if |A| < 1 then // Exploitation: Shrinking circle
                D = |C * X* - X(t)|
                X(t+1) = X* - A * D
            else // Exploration: Search for prey
                X_rand = a randomly selected whale
                D = |C * X_rand - X(t)|
                X(t+1) = X_rand - A * D
            end if
        else
            // --- Exploitation: Spiral Updating ---
            D' = |X* - X(t)|
            b = 1 // Constant defining the shape of the spiral
            l = random number in [-1, 1]
            X(t+1) = D' * exp(b*l) * cos(2*pi*l) + X*
        end if
    end for
    
    t = t + 1
end while

Return X*
```

## 3. Reference

**Title**: "The Whale Optimization Algorithm"
**Authors**: S. Mirjalili, A. Lewis
**URL**: https://www.sciencedirect.com/science/article/pii/S095741741500125X
