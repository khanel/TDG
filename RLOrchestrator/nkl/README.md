# NKL Problem Module

NK-Landscape (NKL) problem implementation with a comprehensive solver pool for RL-based orchestration.

## Solver Pool

**11 Explorers × 13 Exploiters = 143 unique pairings**

### Explorers (Diversity-focused)
| Solver | Base Algorithm | Key Tuning |
|--------|----------------|------------|
| `NKLMapElitesExplorer` | MAP-Elites QD | Archive-based diversity |
| `NKLGWOExplorer` | Grey Wolf Optimizer | Weak hierarchy, high randomness |
| `NKLPSOExplorer` | Particle Swarm | Low inertia, high cognitive |
| `NKLGAExplorer` | Genetic Algorithm | High mutation, low selection |
| `NKLABCExplorer` | Artificial Bee Colony | Aggressive scouting |
| `NKLWOAExplorer` | Whale Optimization | Extended spiral search |
| `NKLHHOExplorer` | Harris Hawks | Soft besiege emphasis |
| `NKLMPAExplorer` | Marine Predators | Lévy flights dominant |
| `NKLSMAExplorer` | Slime Mould | Weak feedback, high randomness |
| `NKLGSAExplorer` | Gravitational Search | Low gravitational constant |
| `NKLDiversityExplorer` | Custom | Explicit diversity maintenance |

### Exploiters (Convergence-focused)
| Solver | Base Algorithm | Key Tuning |
|--------|----------------|------------|
| `NKLBinaryPSOExploiter` | Binary PSO | High velocity, strong attraction |
| `NKLGWOExploiter` | Grey Wolf Optimizer | Strong hierarchy |
| `NKLPSOExploiter` | Particle Swarm | High social component |
| `NKLGAExploiter` | Genetic Algorithm | Elitism, low mutation |
| `NKLLSHADEExploiter` | L-SHADE | Adaptive DE |
| `NKLWOAExploiter` | Whale Optimization | Encircling emphasis |
| `NKLHHOExploiter` | Harris Hawks | Hard besiege emphasis |
| `NKLMPAExploiter` | Marine Predators | Brownian motion dominant |
| `NKLSMAExploiter` | Slime Mould | Strong feedback |
| `NKLGSAExploiter` | Gravitational Search | High gravitational constant |
| `NKLHillClimbingExploiter` | Hill Climbing | Steepest ascent |
| `NKLMemeticExploiter` | Memetic Algorithm | GA + local search |
| `NKLABCExploiter` | Artificial Bee Colony | Greedy selection |

## Usage

```python
from RLOrchestrator.nkl.adapter import NKLAdapter
from RLOrchestrator.nkl.solvers import NKLGWOExplorer, NKLGAExploiter
from RLOrchestrator.core.orchestrator import OrchestratorEnv

# Create problem
problem = NKLAdapter(n_items=100, k_interactions=5)

# Create solvers
explorer = NKLGWOExplorer(problem, population_size=32)
exploiter = NKLGAExploiter(problem, population_size=32)

# Create environment
env = OrchestratorEnv(
    problem=problem,
    exploration_solver=explorer,
    exploitation_solver=exploiter,
    max_decision_steps=50,
    search_steps_per_decision=10,
)
```

## Registry Integration

```python
from RLOrchestrator.problems.registry import instantiate_problem

# Random solver pairing each episode
bundle = instantiate_problem("nkl")
# bundle.stages[0].solver -> random explorer
# bundle.stages[1].solver -> random exploiter
```

## Testing

```bash
python test/nkl/test_solvers.py
# 65 tests covering instantiation, functionality, registry, pipeline
```
