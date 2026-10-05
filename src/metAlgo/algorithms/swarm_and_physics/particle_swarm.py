import numpy as np
from typing import Tuple, Optional
from metAlgo.framework.base_problem import BaseProblem
from metAlgo.framework.base_algorithm import BaseAlgorithm


class ParticleSwarmOptimization(BaseAlgorithm):
    def __init__(
        self,
        problem: BaseProblem,
        pop_size: int = 40,
        w: float = 0.729,      
        c1: float = 1.494,     
        c2: float = 1.494 , *args, **kwargs    
    ):
        super().__init__(problem, name="ParticleSwarmOptimization")
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.pbest_positions: Optional[np.ndarray] = None
        self.pbest_fitness: Optional[np.ndarray] = None

    def initialize(self) -> None:
        if self.problem.is_discrete():
            raise NotImplementedError("Standard PSO is designed for continuous search spaces. It cannot work with the TSP.")

        bounds = self.problem.bounds
        dim = self.problem.dim

        self.positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, dim))
        self.velocities = np.zeros((self.pop_size, dim))
        
        self.fitness = np.array([self.problem.evaluate(pos) for pos in self.positions])

        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.fitness.copy()
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = self.positions[best_idx].copy()
        self.population = self.positions

    def step(self) -> Tuple[np.ndarray, float]:
        bounds = self.problem.bounds
        dim = self.problem.dim
        pop_size = self.pop_size
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)
        cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
        social = self.c2 * r2 * (self.best_solution - self.positions)
        self.velocities = (self.w * self.velocities) + cognitive + social
        self.positions = self.positions + self.velocities
        self.positions = np.clip(self.positions, bounds[:, 0], bounds[:, 1])
        self.fitness = np.array([self.problem.evaluate(p) for p in self.positions])
        improved_mask = self.fitness < self.pbest_fitness
        self.pbest_fitness[improved_mask] = self.fitness[improved_mask]
        self.pbest_positions[improved_mask] = self.positions[improved_mask].copy()
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_solution = self.positions[min_idx].copy()

        return self.best_solution, self.best_fitness
