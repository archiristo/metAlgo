# metAlgo: Advanced Meta-Heuristic Optimization Framework

`metAlgo` is a high-performance and extensible meta-heuristic optimization library developed for academic research and industrial optimization problems.

## 🚀 Why metAlgo?
* **25+ Algorithm Support:** A wide range from Hill Climbing to CMA-ES, NSGA-II to Genocop.
* **Academic Standard Reporting:** Directly transfer your experimental results to your papers with IEEE format LaTeX tables and Friedman statistical analyses.
* **Autonomous Experiment Arena:** Define your problems, provide parameters; the framework tests and reports all algorithms with independent conditions.
* **Secure Architecture:** Type-safe, extensible, and 100% tested core architecture.

## 🛠 Quick Start

```python
from metAlgo.analytics.benchmark_generator import RastriginProblem
from metAlgo.algorithms.swarm_and_physics.particle_swarm import ParticleSwarmOptimization

# Define the problem (10-dimensional Rastrigin)
problem = RastriginProblem(dim=10)

# Configure and run the algorithm
solver = ParticleSwarmOptimization(problem, pop_size=30)
solver.initialize()
best_sol, best_fit = solver.solve()

print(f"Best solution: {best_fit}")
```

## 🏗 Architectural Structure
The framework is built around the BaseAlgorithm and BaseProblem classes, and integrating new algorithms into the system only requires defining a few methods (initialize, step).

## 📚 Documentation
README.md: Installation and development guide.
API_REFERENCE.md: Detailed technical descriptions of classes and functions.

## ⚖️ License
This project is protected under the MIT License. See the LICENSE file for details.
Developed by: archiristo | v1.0 | 2026