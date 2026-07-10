import numpy as np
from typing import Tuple, Optional
from metAlgo.framework.base_problem import BaseProblem
from metAlgo.framework.base_algorithm import BaseAlgorithm


class DifferentialEvolution(BaseAlgorithm):
    def __init__(
        self,
        problem: BaseProblem,
        pop_size: int = 50,
        F: float = 0.5,     
        CR: float = 0.9 , *args, **kwargs      
    ):
        super().__init__(problem, name="DifferentialEvolution")
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.pop: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None

    def initialize(self) -> None:
        if self.problem.is_discrete():
            raise NotImplementedError("The standard DE is for continuous spaces.")

        bounds = self.problem.bounds
        dim = self.problem.dim

        self.population= np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, dim))
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = self.population[best_idx].copy()

    def step(self) -> Tuple[np.ndarray, float]:
        bounds = self.problem.bounds
        dim = self.problem.dim
        new_pop = []
        for i in range(self.pop_size):
            available_indices = [idx for idx in range(self.pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(available_indices, 3, replace=False)
            mutant_vector = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
            mutant_vector = np.clip(mutant_vector, bounds[:, 0], bounds[:, 1])
            trial_vector = self.population[i].copy()
            j_rand = np.random.randint(dim)

            for j in range(dim):
                if np.random.rand() < self.CR or j == j_rand:
                    trial_vector[j] = mutant_vector[j]
            trial_fitness = self.problem.evaluate(trial_vector)
            
            if trial_fitness < self.fitness[i]:
                new_pop.append(trial_vector)
                self.fitness[i] = trial_fitness
            else:
                new_pop.append(self.population[i].copy())
        self.population= np.array(new_pop)
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()

        return self.best_solution, self.best_fitness
