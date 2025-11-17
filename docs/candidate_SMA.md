# Candidate Algorithm: Slime Mould Algorithm (SMA)

## 1. High-Level Description

**Slime Mould Algorithm (SMA)** is a metaheuristic algorithm inspired by the dynamic foraging behavior of slime mould (*Physarum polycephalum*). The algorithm simulates the mould's ability to find optimal paths between food sources by modeling its propagation and contraction phases.

The core of SMA lies in its unique **adaptive weighting mechanism**. The algorithm simulates the positive and negative feedback responses of the slime mould's veins as it approaches or moves away from high-quality food. This creates a distinctive oscillating behavior that naturally balances exploration and exploitation:

*   When a food source is of high quality, the weight of the path leading to it increases, encouraging more individuals (mould) to move towards it (exploitation).
*   When a food source is of low quality, the weight of the path decreases, causing individuals to search for alternative routes (exploration).

This process allows the algorithm to dynamically adapt its search strategy based on the quality of the solutions it has found so far.

## 2. Pseudocode

```
Initialize the population of slime mould individuals X_i
Set max_iterations T

while (t < T) do
    // Evaluate fitness for all individuals
    // Update bestFitness, worstFitness, and bestPosition
    
    for each individual X_i do
        // Calculate the weight W for the current individual
        // This is based on its fitness relative to the best and worst in the population
        W_i = calculate_weight(fitness(X_i), bestFitness, worstFitness)
        
        // Update parameters p and vb
        p = tanh|fitness(X_i) - bestFitness|
        vc = value that linearly decreases from 1 to 0
        
        // Update position
        if rand() < p then
            // Randomly search the space
            X(t+1) = rand() * (UpperBound - LowerBound) + LowerBound
        else
            // Select two random individuals A and B from the population
            X_A = population[rand_index_A]
            X_B = population[rand_index_B]
            
            r = rand()
            if r < vc then
                // Move towards the best position
                X(t+1) = bestPosition + vb * (W_i * X_A - X_B)
            else
                // Circle the current position
                X(t+1) = vc * X(t)
            end if
        end if
    end for
    
    t = t + 1
end while

Return bestPosition
```

## 3. Reference

**Title**: "Slime mould algorithm: A new method for stochastic optimization"
**Authors**: S. Li, H. Chen, M. Wang, A. Heidari, S. Mirjalili
**URL**: https://ieeexplore.ieee.org/document/9024269
