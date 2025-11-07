# Product Context

## Problem

Meta-heuristic solvers often have different strengths. Some are excellent at exploring a vast search space to find promising regions, while others are skilled at exploiting those regions to find high-quality solutions. Manually tuning when to switch from an exploration-focused solver to an exploitation-focused one is difficult, time-consuming, and often sub-optimal.

## Solution

This project aims to create a "problem-agnostic RL orchestrator" that learns the optimal time to transition between search stages. By treating the "when to switch" decision as a reinforcement learning problem, the system can develop a policy that outperforms fixed or manually-tuned strategies.

## User Experience

The intended user is a researcher or engineer who wants to solve complex optimization problems without needing to become an expert in the fine-tuning of meta-heuristic pipelines. The orchestrator should be configurable for different problems (TSP, MaxCut, etc.) and solver combinations, providing a general-purpose tool for automated algorithm configuration.