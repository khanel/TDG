#!/usr/bin/env python3
"""
Integration tests for NKL solvers with the OrchestratorEnv pipeline.

These tests verify that:
1. All explorer/exploiter classes can be instantiated
2. Solvers integrate properly with NKLAdapter
3. The full orchestrator pipeline runs without errors
4. Registry correctly randomizes solver pairings
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RLOrchestrator.nkl.adapter import NKLAdapter
from RLOrchestrator.nkl.solvers import (
    EXPLORER_CLASSES,
    EXPLOITER_CLASSES,
    # Explorers
    NKLMapElitesExplorer,
    NKLGWOExplorer,
    NKLPSOExplorer,
    NKLGAExplorer,
    NKLABCExplorer,
    NKLWOAExplorer,
    NKLHHOExplorer,
    NKLMPAExplorer,
    NKLSMAExplorer,
    NKLGSAExplorer,
    NKLDiversityExplorer,
    # Exploiters
    NKLBinaryPSOExploiter,
    NKLGWOExploiter,
    NKLPSOExploiter,
    NKLGAExploiter,
    NKLLSHADEExploiter,
    NKLWOAExploiter,
    NKLHHOExploiter,
    NKLMPAExploiter,
    NKLSMAExploiter,
    NKLGSAExploiter,
    NKLHillClimbingExploiter,
    NKLMemeticExploiter,
    NKLABCExploiter,
)
from RLOrchestrator.problems.registry import instantiate_problem, get_problem_definition
from RLOrchestrator.core.orchestrator import OrchestratorEnv


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def nkl_problem():
    """Create a small NKL problem for testing."""
    return NKLAdapter(n_items=20, k_interactions=3, seed=42)


@pytest.fixture
def nkl_problem_medium():
    """Create a medium NKL problem for testing."""
    return NKLAdapter(n_items=50, k_interactions=5, seed=42)


# =============================================================================
# Test Solver Instantiation
# =============================================================================

class TestSolverInstantiation:
    """Test that all solvers can be instantiated."""
    
    def test_explorer_count(self):
        """Verify we have 11 explorer classes."""
        assert len(EXPLORER_CLASSES) == 11, f"Expected 11 explorers, got {len(EXPLORER_CLASSES)}"
    
    def test_exploiter_count(self):
        """Verify we have 13 exploiter classes."""
        assert len(EXPLOITER_CLASSES) == 13, f"Expected 13 exploiters, got {len(EXPLOITER_CLASSES)}"
    
    @pytest.mark.parametrize("explorer_cls", EXPLORER_CLASSES)
    def test_explorer_instantiation(self, nkl_problem, explorer_cls):
        """Test that each explorer can be instantiated with default params."""
        explorer = explorer_cls(nkl_problem, population_size=10)
        assert explorer is not None
        assert hasattr(explorer, 'step')
        assert hasattr(explorer, 'get_best')
    
    @pytest.mark.parametrize("exploiter_cls", EXPLOITER_CLASSES)
    def test_exploiter_instantiation(self, nkl_problem, exploiter_cls):
        """Test that each exploiter can be instantiated with default params."""
        exploiter = exploiter_cls(nkl_problem, population_size=10)
        assert exploiter is not None
        assert hasattr(exploiter, 'step')
        assert hasattr(exploiter, 'get_best')


# =============================================================================
# Test Solver Functionality
# =============================================================================

class TestSolverFunctionality:
    """Test that solvers produce valid solutions."""
    
    @pytest.mark.parametrize("explorer_cls", EXPLORER_CLASSES)
    def test_explorer_produces_solutions(self, nkl_problem, explorer_cls):
        """Test that explorers produce valid binary solutions."""
        explorer = explorer_cls(nkl_problem, population_size=10)
        if hasattr(explorer, 'initialize'):
            explorer.initialize()
        
        # Run a few steps
        for _ in range(5):
            explorer.step()
        
        best = explorer.get_best()
        assert best is not None, f"{explorer_cls.__name__} returned None best"
        assert hasattr(best, 'representation'), f"{explorer_cls.__name__} best has no representation"
        assert hasattr(best, 'fitness'), f"{explorer_cls.__name__} best has no fitness"
        
        # Verify solution is binary
        solution = best.representation
        dim = nkl_problem.nkl_problem.n
        assert all(x in (0, 1) for x in solution), f"{explorer_cls.__name__} produced non-binary solution"
        assert len(solution) == dim, f"{explorer_cls.__name__} solution wrong dimension"
    
    @pytest.mark.parametrize("exploiter_cls", EXPLOITER_CLASSES)
    def test_exploiter_produces_solutions(self, nkl_problem, exploiter_cls):
        """Test that exploiters produce valid binary solutions."""
        exploiter = exploiter_cls(nkl_problem, population_size=10)
        if hasattr(exploiter, 'initialize'):
            exploiter.initialize()
        
        # Run a few steps
        for _ in range(5):
            exploiter.step()
        
        best = exploiter.get_best()
        assert best is not None, f"{exploiter_cls.__name__} returned None best"
        assert hasattr(best, 'representation'), f"{exploiter_cls.__name__} best has no representation"
        assert hasattr(best, 'fitness'), f"{exploiter_cls.__name__} best has no fitness"
        
        # Verify solution is binary
        solution = best.representation
        dim = nkl_problem.nkl_problem.n
        assert all(x in (0, 1) for x in solution), f"{exploiter_cls.__name__} produced non-binary solution"
        assert len(solution) == dim, f"{exploiter_cls.__name__} solution wrong dimension"


# =============================================================================
# Test Registry Integration
# =============================================================================

class TestRegistryIntegration:
    """Test solver integration with the problem registry."""
    
    def test_nkl_registered(self):
        """Test that NKL problem is registered."""
        definition = get_problem_definition("nkl")
        assert definition is not None, "NKL problem not registered"
        assert definition.name == "nkl"
    
    def test_nkl_metadata(self):
        """Test NKL problem metadata."""
        definition = get_problem_definition("nkl")
        metadata = definition.metadata
        assert metadata.get("explorer_count") == 11
        assert metadata.get("exploiter_count") == 13
        assert metadata.get("total_pairings") == 143
    
    def test_instantiate_nkl(self):
        """Test that NKL problem can be instantiated from registry."""
        bundle = instantiate_problem("nkl")
        assert bundle is not None
        assert bundle.name == "nkl"
        assert bundle.problem is not None
        assert len(bundle.stages) == 2
    
    def test_solver_randomization(self):
        """Test that registry randomizes solver selection."""
        # Instantiate multiple times and collect solver class names
        explorer_classes = set()
        exploiter_classes = set()
        
        for _ in range(50):
            bundle = instantiate_problem("nkl")
            for stage in bundle.stages:
                if stage.name == "exploration":
                    explorer_classes.add(type(stage.solver).__name__)
                elif stage.name == "exploitation":
                    exploiter_classes.add(type(stage.solver).__name__)
        
        # Should see multiple different solver types (with high probability)
        # With 50 samples from 11 explorers, we expect to see at least 3 different ones
        assert len(explorer_classes) >= 3, f"Expected variety in explorers, got: {explorer_classes}"
        assert len(exploiter_classes) >= 3, f"Expected variety in exploiters, got: {exploiter_classes}"


# =============================================================================
# Test Full Pipeline Integration
# =============================================================================

class TestPipelineIntegration:
    """Test full orchestrator pipeline with NKL solvers."""
    
    def test_env_creation(self, nkl_problem):
        """Test OrchestratorEnv creation with NKL."""
        explorer = NKLGWOExplorer(nkl_problem, population_size=16)
        exploiter = NKLGAExploiter(nkl_problem, population_size=16)
        
        env = OrchestratorEnv(
            problem=nkl_problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=20,
            search_steps_per_decision=5,
        )
        
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None
    
    def test_env_reset(self, nkl_problem):
        """Test environment reset."""
        explorer = NKLMapElitesExplorer(nkl_problem, population_size=16)
        exploiter = NKLBinaryPSOExploiter(nkl_problem, population_size=16)
        
        env = OrchestratorEnv(
            problem=nkl_problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=20,
            search_steps_per_decision=5,
        )
        
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert len(obs) == 6, f"Expected 6 observation features, got {len(obs)}"
        assert info is not None
    
    def test_env_step(self, nkl_problem):
        """Test environment step."""
        explorer = NKLPSOExplorer(nkl_problem, population_size=16)
        exploiter = NKLHillClimbingExploiter(nkl_problem, population_size=16)
        
        env = OrchestratorEnv(
            problem=nkl_problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=20,
            search_steps_per_decision=5,
        )
        
        obs, info = env.reset(seed=42)
        
        # Take a STAY action
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_full_episode(self, nkl_problem):
        """Test running a full episode."""
        explorer = NKLABCExplorer(nkl_problem, population_size=16)
        exploiter = NKLMemeticExploiter(nkl_problem, population_size=16)
        
        env = OrchestratorEnv(
            problem=nkl_problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=30,
            search_steps_per_decision=5,
        )
        
        obs, info = env.reset(seed=42)
        done = False
        total_steps = 0
        total_reward = 0.0
        
        while not done and total_steps < 100:
            # Simple policy: stay until stagnation > 0.5, then advance
            stagnation = obs[3]  # stagnation is at index 3
            action = 1 if stagnation > 0.5 else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
            total_reward += reward
        
        assert done, "Episode should terminate"
        assert total_steps > 0, "Episode should have at least one step"
        env.close()
    
    @pytest.mark.parametrize("explorer_cls,exploiter_cls", [
        (NKLMapElitesExplorer, NKLBinaryPSOExploiter),
        (NKLGWOExplorer, NKLGAExploiter),
        (NKLHHOExplorer, NKLLSHADEExploiter),
        (NKLMPAExplorer, NKLMemeticExploiter),
        (NKLSMAExplorer, NKLHillClimbingExploiter),
    ])
    def test_various_solver_pairs(self, nkl_problem, explorer_cls, exploiter_cls):
        """Test various explorer/exploiter combinations."""
        explorer = explorer_cls(nkl_problem, population_size=12)
        exploiter = exploiter_cls(nkl_problem, population_size=12)
        
        env = OrchestratorEnv(
            problem=nkl_problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=15,
            search_steps_per_decision=3,
        )
        
        obs, _ = env.reset(seed=42)
        
        # Run 10 steps with alternating actions
        for i in range(10):
            action = i % 2
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        env.close()


# =============================================================================
# Test Observation Features
# =============================================================================

class TestObservationFeatures:
    """Test that observation features are computed correctly."""
    
    def test_observation_bounds(self, nkl_problem):
        """Test that observations are within expected bounds."""
        explorer = NKLDiversityExplorer(nkl_problem, population_size=16)
        exploiter = NKLGSAExploiter(nkl_problem, population_size=16)
        
        env = OrchestratorEnv(
            problem=nkl_problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=20,
            search_steps_per_decision=5,
        )
        
        obs, _ = env.reset(seed=42)
        
        # Run several steps and check bounds
        for _ in range(15):
            # All observation features should be in [0, 1]
            for i, val in enumerate(obs):
                assert 0.0 <= val <= 1.0, f"Observation feature {i} = {val} out of bounds"
            
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        env.close()
    
    def test_phase_encoding(self, nkl_problem):
        """Test phase encoding in observations."""
        explorer = NKLWOAExplorer(nkl_problem, population_size=16)
        exploiter = NKLWOAExploiter(nkl_problem, population_size=16)
        
        env = OrchestratorEnv(
            problem=nkl_problem,
            exploration_solver=explorer,
            exploitation_solver=exploiter,
            max_decision_steps=20,
            search_steps_per_decision=5,
        )
        
        obs, _ = env.reset(seed=42)
        
        # Phase is at index 5
        phase = obs[5]
        # Should start in exploration (phase = 0.0)
        assert phase == 0.0, f"Expected exploration phase (0.0), got {phase}"
        
        # Advance to exploitation
        obs, _, _, _, _ = env.step(1)  # ADVANCE
        phase = obs[5]
        # Should be in exploitation (phase = 0.5)
        assert phase == 0.5, f"Expected exploitation phase (0.5), got {phase}"
        
        env.close()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
