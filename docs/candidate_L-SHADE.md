# Candidate Algorithm: L-SHADE

## 1. High-Level Description

**L-SHADE (Success-History based Adaptive DE with Linear Population Size Reduction)** is a powerful and modern variant of the Differential Evolution (DE) algorithm. It is widely recognized as a top-performing algorithm for global optimization problems, having won several IEEE CEC (Congress on Evolutionary Computation) competitions.

The core innovation of L-SHADE is its **self-adaptation mechanism**. Instead of requiring the user to fine-tune the sensitive control parameters of DE (the scaling factor `F` and crossover rate `CR`), L-SHADE automatically adjusts them during the run. It maintains a historical "memory" of parameter values that have led to successful solutions in the past and samples new parameters from a distribution biased towards these successful values.

Additionally, it features a simple but effective mechanism for **linear population size reduction**, which gradually reduces the number of solutions in the population, helping to focus computational effort as the search progresses.

## 2. Pseudocode

```
Initialize population P of size NP
Initialize an empty archive A
Initialize historical memory M_F and M_CR for scaling factor and crossover rate

for generation g = 1 to max_generations do
    S_F = empty set
    S_CR = empty set

    for each individual x_i in P do
        // Generate adaptive parameters for this individual
        r = select random index from [1, |M_CR|]
        CR_i = sample from Normal distribution with mean M_CR[r] and stddev 0.1
        F_i = sample from Cauchy distribution with location M_F[r] and scale 0.1
        
        // Mutation Strategy (e.g., 'current-to-pbest/1')
        p_best = select one of the top p% best individuals from P randomly
        x_r1 = select random individual from P
        x_r2 = select random individual from P union A
        
        v_i = x_i + F_i * (p_best - x_i) + F_i * (x_r1 - x_r2) // Trial vector
        
        // Crossover
        u_i = crossover(x_i, v_i, CR_i)
        
        // Selection
        if fitness(u_i) < fitness(x_i) then
            P_next_gen[i] = u_i
            add x_i to archive A
            add F_i to S_F
            add CR_i to S_CR
        else
            P_next_gen[i] = x_i
        end if
    end for
    
    // Update historical memory
    if S_F is not empty then
        update M_F and M_CR using weighted means of values in S_F and S_CR
    end if
    
    // Update population and archive
    P = P_next_gen
    if |A| > NP then
        remove random individuals from A until |A| = NP
    end if
    
    // Linear Population Size Reduction
    NP_next = round(NP_initial - g * (NP_initial - NP_min) / max_generations)
    if NP_next < NP then
        remove (NP - NP_next) worst individuals from P
        NP = NP_next
    end if
    
end for

Return the best individual found in P
```

## 3. Reference

**Title**: "Success-History Based Parameter Adaptation for Differential Evolution"
**Authors**: J. Tanabe and A. Fukunaga
**URL**: https://ieeexplore.ieee.org/document/6557555
