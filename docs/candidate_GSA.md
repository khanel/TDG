# Candidate Algorithm: Gravitational Search Algorithm (GSA)

## 1. High-Level Description

The **Gravitational Search Algorithm (GSA)** is a physics-inspired metaheuristic based on Newton's laws of gravity and motion. In this algorithm, the candidate solutions are treated as "agents" or "masses" interacting with each other in a simulated universe.

The core principles are:

*   **Mass as Fitness**: The "mass" of each agent is directly proportional to its fitness. Better solutions are heavier and have a stronger gravitational pull.
*   **Gravitational Force**: Agents exert gravitational forces on one another. A heavier agent (better solution) will pull other, lighter agents towards it. The total force on an agent is the sum of the forces exerted by other agents (often limited to a subset of the best agents to reduce computational cost).
*   **Law of Motion**: An agent's movement (its acceleration and velocity) is determined by the net force acting upon it. This causes agents to accelerate towards regions of the search space with higher mass concentration, effectively exploring and exploiting promising areas.

The gravitational constant `G` is initialized at a high value and decreases over time. This allows for a wide-ranging, explorative search in the beginning (high `G`) and a more focused, exploitative search in the end (low `G`).

## 2. Pseudocode

```
Initialize the population of agents (solutions) X_i with random positions
Set max_iterations T

while (t < T) do
    // Evaluate fitness for all agents
    // Update best and worst fitness in the population
    
    // Update Gravitational Constant G(t)
    G(t) = G_0 * exp(-alpha * t/T)
    
    // Calculate Mass M for each agent
    for each agent i do
        m_i = (fitness(i) - worst) / (best - worst)
    end for
    for each agent i do
        M_i = m_i / sum(m)
    end for
    
    // Calculate Total Force F
    for each agent i do
        F_i = vector of zeros
        // Consider force from the k-best agents only
        k_best = a subset of agents with the best fitness
        for each agent j in k_best where j != i do
            F_ij = G(t) * (M_i * M_j) / (distance(i, j) + epsilon) * (X_j - X_i)
            F_i = F_i + rand() * F_ij
        end for
    end for
    
    // Calculate Acceleration and Update Velocity/Position
    for each agent i do
        a_i(t) = F_i / M_i // Acceleration
        v_i(t+1) = rand() * v_i(t) + a_i(t) // Velocity
        X_i(t+1) = X_i(t) + v_i(t+1) // Position
    end for
    
    t = t + 1
end while

Return the position of the agent with the best fitness
```

## 3. Reference

**Title**: "GSA: A Gravitational Search Algorithm"
**Authors**: E. Rashedi, H. Nezamabadi-pour, S. Saryazdi
**URL**: https://www.sciencedirect.com/science/article/pii/S095741740900086X
