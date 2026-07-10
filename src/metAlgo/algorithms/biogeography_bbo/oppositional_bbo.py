import numpy as np
from typing import Tuple, Optional
from metAlgo.framework.base_problem import BaseProblem
from metAlgo.algorithms.biogeography_bbo.bbo_base import BiogeographyBasedOptimization


class OppositionalBBO(BiogeographyBasedOptimization):
    def __init__(
        self,
        problem: BaseProblem,
        pop_size: int = 50,
        mutation_rate_max: float = 0.01, 
        jr: float = 0.35, *args, **kwargs
    ) -> None:
        super().__init__(
            problem, 
            pop_size=pop_size, 
            mutation_rate_max=mutation_rate_max
        )
        self.jr = jr
        self.name = "OppositionalBBO"

    def _compute_opposite_points(self, points: np.ndarray) -> np.ndarray:
        current_min = np.min(points, axis=0)
        current_max = np.max(points, axis=0)
        
        opposite = current_min + current_max - points
        bounds = self.problem.bounds
        opposite = np.clip(opposite, bounds[:, 0], bounds[:, 1])
        return opposite

    def initialize(self) -> None:
        super().initialize()
        opposite_pop = self._compute_opposite_points(self.population)
        opposite_fitness = np.array([self.problem.evaluate(ind) for ind in opposite_pop])
        
        combined_pop = np.vstack((self.population, opposite_pop))
        combined_fitness = np.concatenate((self.fitness, opposite_fitness))
        
        best_indices = np.argsort(combined_fitness)[:self.pop_size]
        self.population= combined_pop[best_indices]
        self.fitness = combined_fitness[best_indices]
        self.best_fitness = self.fitness[0]
        self.best_solution = self.population[0].copy()

    def step(self) -> Tuple[np.ndarray, float]:
        best_sol, best_fit = super().step()
        if np.random.rand() < self.jr:
            opposite_pop = self._compute_opposite_points(self.population)
            opposite_fitness = np.array([self.problem.evaluate(ind) for ind in opposite_pop])
            
            combined_pop = np.vstack((self.population, opposite_pop))
            combined_fitness = np.concatenate((self.fitness, opposite_fitness))
            
            best_indices = np.argsort(combined_fitness)[:self.pop_size]
            self.population= combined_pop[best_indices]
            self.fitness = combined_fitness[best_indices]
            
            if self.fitness[0] < self.best_fitness:
                self.best_fitness = self.fitness[0]
                self.best_solution = self.population[0].copy()
                
        return self.best_solution, self.best_fitness
