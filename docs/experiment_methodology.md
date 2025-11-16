# Experiment Methodology

## 1. Guiding Principles

The experimental process is designed to be incremental, systematic, and rigorous. The primary goal is to isolate the impact of individual features on the performance of the RL orchestrator. All experiments will follow a strict one-feature-at-a-time addition to a stable baseline.

## 2. The Baseline: 6D Observation Space

All experiments will start from a common baseline: the 6-dimensional observation space. This baseline provides the agent with essential, high-level information about the search process.

The 6D observation space consists of:
1.  `budget_remaining`
2.  `normalized_best_fitness`
3.  `improvement_velocity`
4.  `stagnation`
5.  `population_diversity`
6.  `active_phase`

## 3. Prerequisite: A Diverse Pool of Configured Solvers

A core assumption of the generalized training approach is the availability of a **diverse pool of solvers** for each problem. The agent's ability to learn a robust, general-purpose policy is directly dependent on being exposed to different search behaviors during training. A single exploration and exploitation solver per problem is insufficient.

Therefore, a critical prerequisite for any experimentation is to ensure that a variety of solvers, each representing a different search strategy (e.g., evolutionary, swarm-based, local search), are implemented and correctly configured.

### a. Solver Configuration Requirements
-   **Phase Attribute:** Every solver class **must** have a `phase` class attribute set to either `'exploration'` or `'exploitation'`. This is how the dynamic registry discovers and categorizes them.
-   **Standardized Parameters:** For the initial baseline experiments, all solvers of the same type should use a standardized set of default hyperparameters to ensure a fair comparison.

### b. Confirmed Solver Pool
The following solvers are confirmed to be implemented and correctly configured with a `phase` attribute:

-   **TSP (Traveling Salesperson Problem):**
    -   *Exploration:* `TSPMapElites`
    -   *Exploitation:* `TSPParticleSwarm`
-   **Knapsack:**
    -   *Exploration:* `KnapsackRandomExplorer`
    -   *Exploitation:* `KnapsackBitFlipExploiter`
-   **MaxCut:**
    -   *Exploration:* `MaxCutRandomExplorer`
    -   *Exploitation:* `MaxCutBitFlipExploiter`
-   **NKL (NK-Landscape):**
    -   *Exploration:* `NKLRandomExplorer`
    -   *Exploitation:* `NKLBitFlipExploiter`

## 4. Experimental Process

### Step 1: Baseline Evaluation
The first step is to comprehensively test the performance of the policy trained on the baseline 6D observation space.
-   **Initial Problems:** MaxCut and NKL.
-   **Full Evaluation:** After the initial tests, the evaluation will be expanded to include TSP.
-   **Output:** All results will be meticulously documented.

### Step 2: Incremental Feature Addition
-   **One Feature at a Time:** For each subsequent experiment, **only one** new feature will be added to the 6D baseline, creating a 7D observation space.
-   **No Compounding Features:** We will not create 8D or higher-dimensional spaces in this phase. Each experiment tests the 6D baseline against a `6D + 1 new feature` configuration.
-   **Training and Testing:** The new 7D policy will be trained and evaluated using the same comprehensive protocol as the baseline.

### Step 3: Analysis and Selection
After this series of experiments, the results will be analyzed to identify the most promising features. This will inform the selection of a "first filtered observation space candidate" for future, more complex experiments.

## 4. Training Protocol

To ensure that the learned policies are robust and the comparisons are fair, a strict training protocol will be followed for all experiments.

### a. Generalized Training
The agent will be trained on a variety of problems and solver combinations simultaneously. This "generalized" approach is designed to produce a single, robust policy that can adapt to different contexts, rather than overfitting to a specific problem instance.

### b. Key Training Parameters
-   **Total Timesteps:** A substantial number of training timesteps (e.g., 2,000,000 or more) will be used to ensure the agent has sufficient experience to learn a meaningful policy.
-   **Parallel Environments:** Training will be parallelized across multiple environments (e.g., 4 or more) to stabilize and accelerate the learning process. Each environment will be seeded with a different random problem instance and solver combination.
-   **Algorithm:** The Proximal Policy Optimization (PPO) algorithm will be used, as it is a well-established, state-of-the-art algorithm for this type of problem.
-   **Hyperparameters:** The default hyperparameters from a reputable library (e.g., Stable Baselines3) will be used as a starting point. Any changes to these hyperparameters will be documented and justified.

### c. Training Correctness and Validation
-   **Seed Management:** All random number generators will be carefully seeded to ensure reproducibility.
-   **Logging and Monitoring:** The training process will be closely monitored, with key metrics (e.g., reward, episode length) logged to identify any anomalies or convergence issues.
-   **Policy Saving:** The final trained policy for each experiment will be saved to a unique file (e.g., `ppo_6d_baseline.zip`, `ppo_7d_funnel_proxy.zip`) for easy identification and use in the evaluation phase.

## 5. Metrics and Reporting

### a. Comprehensive Metrics
The evaluation must be comprehensive, capturing both the **effectiveness** (quality of the final solution) and **efficiency** (resources used to get there) of the policy.

Metrics must include, but are not limited to:
-   **Final Best Fitness:** The primary measure of solution quality.
-   **Time to Best:** The wall-clock time or number of steps required to find the best solution.
-   **Convergence Curves:** Plots showing fitness improvement over time.
-   **Switch Step:** The decision step at which the policy chose to `ADVANCE`.
-   **Resource Utilization:** CPU time, memory usage, etc.

### b. Detailed Reporting
-   **Raw Data:** All raw results must be saved in a structured, machine-readable format (e.g., CSV).
-   **Analysis and Summaries:** The results of each experiment must be analyzed, with key findings, charts, and conclusions documented in Markdown files.
-   **Traceability:** The documentation must be clear enough for an external reader to follow the entire experimental process, understand the results, and potentially conduct further analysis.
