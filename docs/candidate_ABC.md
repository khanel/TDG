# Candidate Algorithm: Artificial Bee Colony (ABC)

## 1. High-Level Description

**Artificial Bee Colony (ABC)** is a swarm intelligence algorithm that models the intelligent foraging behavior of a honey bee colony. The algorithm is known for its simplicity, flexibility, and robust performance. The colony of artificial bees is divided into three distinct groups, each with a specific role that contributes to balancing exploration and exploitation:

1.  **Employed Bees**: These bees are associated with a specific "food source" (a candidate solution). They perform a local search by exploring the neighborhood of their assigned source. If they find a better source, they memorize it and forget the old one.

2.  **Onlooker Bees**: These bees wait in the hive and observe the "dances" of the employed bees to choose a food source. They probabilistically select sources to exploit based on their quality (fitness) — richer sources are more likely to be chosen. This introduces a selection pressure that favors better solutions.

3.  **Scout Bees**: This is the algorithm's primary exploration mechanism. If a food source cannot be improved after a predetermined number of attempts (the "limit" parameter), it is considered exhausted and is abandoned. The employed bee associated with that source becomes a scout and begins a random search for a completely new food source anywhere in the search space.

This division of labor creates a robust search dynamic where good solutions are continuously refined by employed and onlooker bees, while the scout bees prevent the algorithm from getting permanently stuck in local optima.

## 2. Pseudocode

```
Initialize population of N food sources (solutions) randomly
Set max_iterations T
Set "limit" for scout bee generation

while (t < T) do
    
    // --- Employed Bee Phase ---
    for each employed bee i do
        // Produce a new candidate solution v_i in the neighborhood of its current source x_i
        v_i = generate_neighbor(x_i)
        
        // Greedy selection
        if fitness(v_i) is better than fitness(x_i) then
            x_i = v_i
            reset trial_counter for x_i
        else
            increment trial_counter for x_i
        end if
    end for
    
    // --- Onlooker Bee Phase ---
    // Calculate probabilities p_i for each source x_i based on its fitness
    
    for each onlooker bee j do
        // Select a source x_i based on probability p_i (e.g., using roulette wheel selection)
        
        // Produce a new candidate solution v_i in the neighborhood of the selected source x_i
        v_i = generate_neighbor(x_i)
        
        // Greedy selection
        if fitness(v_i) is better than fitness(x_i) then
            x_i = v_i
            reset trial_counter for x_i
        else
            increment trial_counter for x_i
        end if
    end for
    
    // --- Scout Bee Phase ---
    for each source x_i do
        if trial_counter for x_i > limit then
            // Abandon the source and replace it with a new random source
            x_i = generate_random_source()
            reset trial_counter for x_i
        end if
    end for
    
    // Memorize the best solution found so far
    
    t = t + 1
end while

Return the best solution found
```

## 3. Reference

**Title**: "An idea based on honey bee swarm for numerical optimization"
**Author**: D. Karaboga
**URL**: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.135.2957 (Link to a technical report of the foundational work)
