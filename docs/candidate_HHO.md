# Candidate Algorithm: Harris Hawks Optimization (HHO)

## 1. High-Level Description

**Harris Hawks Optimization (HHO)** is a swarm-based, nature-inspired optimization algorithm that mimics the cooperative hunting behavior of Harris's hawks. The algorithm is notable for its dynamic and multi-phase structure, which explicitly models the transition from exploration to exploitation.

The core of the algorithm revolves around the "escaping energy" of the prey, a parameter `E` that decreases over the course of the iterations. This energy level determines the behavior of the hawks (the solutions):

*   **Exploration Phase (`|E| >= 1`)**: When the prey has high energy, the hawks are in a wide-ranging search mode. They perch at random locations to scan for prey, waiting to detect a target. This phase promotes global search and diversity.
*   **Exploitation Phase (`|E| < 1`)**: When the prey is tired (low energy), the hawks perform a surprise pounce. This phase is further divided into four distinct attack strategies based on the prey's chance of escaping. These strategies model different types of local search, from a gentle "soft besiege" to an aggressive "hard besiege" and "rapid dives".

This built-in, phased search mechanism makes HHO a powerful standalone solver and an excellent candidate for a framework that values the balance between exploration and exploitation.

## 2. Pseudocode

```
Initialize the population of hawks (solutions) X_i (i = 1, 2, ..., N)
Set max_iterations T

while (t < T) do
    // Evaluate fitness for all hawks and set X_rabbit as the best location (solution)
    
    for each hawk (X_i) do
        // Update the escaping energy of the prey
        E_0 = random number in (-1, 1)
        E = 2 * E_0 * (1 - t/T)
        
        // --- Exploration Phase ---
        if |E| >= 1 then
            q = random number in (0, 1)
            if q >= 0.5 then // Perch based on other family members
                X_rand = random hawk from the current population
                X(t+1) = X_rand - rand() * |X_rand - 2 * rand() * X(t)|
            else // Perch on a random tall tree
                X_m = average position of the current population
                X(t+1) = (X_rabbit - X_m) - rand() * (LowerBound + rand() * (UpperBound - LowerBound))
            end if
        end if
        
        // --- Exploitation Phase ---
        if |E| < 1 then
            r = random number in (0, 1) // Chance of prey successfully escaping
            
            // Soft Besiege
            if r >= 0.5 and |E| >= 0.5 then
                Jump_strength = 2 * (1 - rand())
                X(t+1) = (X_rabbit - X(t)) - E * |Jump_strength * X_rabbit - X(t)|
            end if
            
            // Hard Besiege
            if r >= 0.5 and |E| < 0.5 then
                X(t+1) = X_rabbit - E * |X_rabbit - X(t)|
            end if
            
            // Soft Besiege with Rapid Dives
            if r < 0.5 and |E| >= 0.5 then
                // Perform a dive using Lévy flight (LF)
                Y = X_rabbit - E * |Jump_strength * X_rabbit - X(t)|
                Z = Y + rand(1, D) * LF(D) // D is dimension
                if fitness(Y) < fitness(X(t)) then X(t+1) = Y
                if fitness(Z) < fitness(X(t)) then X(t+1) = Z
            end if
            
            // Hard Besiege with Rapid Dives
            if r < 0.5 and |E| < 0.5 then
                // Perform a dive using Lévy flight (LF)
                Y = X_rabbit - E * |Jump_strength * X_rabbit - X_m|
                Z = Y + rand(1, D) * LF(D)
                if fitness(Y) < fitness(X(t)) then X(t+1) = Y
                if fitness(Z) < fitness(X(t)) then X(t+1) = Z
            end if
        end if
    end for
    
    t = t + 1
end while

Return X_rabbit
```

## 3. Reference

**Title**: "Harris hawks optimization: Algorithm and applications"
**Authors**: A. Heidari, S. Mirjalili, H. Faris, I. Aljarah, M. Mafarja, H. Chen
**URL**: https://onlinelibrary.wiley.com/doi/full/10.1002/int.22177
