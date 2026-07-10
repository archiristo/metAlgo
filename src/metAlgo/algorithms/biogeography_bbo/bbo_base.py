
import numpy as np
from typing import Tuple, Optional
from metAlgo.framework.base_problem import BaseProblem
from metAlgo.framework.base_algorithm import BaseAlgorithm


class BiogeographyBasedOptimization(BaseAlgorithm):


    def __init__(
        self,
        problem: BaseProblem,
        pop_size: int = 50,
        p_mod: float = 1.0,       
        mutation_rate_max: float = 0.01, 
        *args, **kwargs
    ):
        super().__init__(problem, name="BiogeographyBasedOptimization")
        self.pop_size = pop_size
        self.p_mod = p_mod
        self.mutation_rate_max = mutation_rate_max

    
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        
    
        self.lambda_rates = np.zeros(pop_size)
        self.mu_rates = np.zeros(pop_size)
        self.species_count_prob = np.zeros(pop_size) 

    def initialize(self) -> None:
        if self.problem.is_discrete():
            raise NotImplementedError("Standard BBO is for continuous spaces.")

        bounds = self.problem.bounds
        dim = self.problem.dim

        self.population= np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, dim))
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])

        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = self.population[best_idx].copy()

    def _update_migration_rates(self) -> None:
        sorted_indices = np.argsort(self.fitness)[::-1]
        for rank, idx in enumerate(sorted_indices):
            self.lambda_rates[idx] = 1.0 - (rank / (self.pop_size - 1))
            self.mu_rates[idx] = rank / (self.pop_size - 1)

    def step(self) -> Tuple[np.ndarray, float]:
        bounds = self.problem.bounds
        dim = self.problem.dim
        self._update_migration_rates()
        current_best_idx = np.argmin(self.fitness)
        elite_habitat = self.population[current_best_idx].copy()
        
        new_pop = self.population.copy()
        for i in range(self.pop_size):
            if np.random.rand() < self.p_mod * self.lambda_rates[i]:
                mu_sum = np.sum(self.mu_rates)
                if mu_sum > 0:
                    probabilities = self.mu_rates / mu_sum
                    for j in range(dim):
                        if np.random.rand() < self.lambda_rates[i]:
                            selected_island = np.random.choice(self.pop_size, p=probabilities)
                            new_pop[i, j] = self.population[selected_island, j]

            for j in range(dim):
                current_mut_rate = self.mutation_rate_max * self.lambda_rates[i]
                if np.random.rand() < current_mut_rate:
                    new_pop[i, j] = np.random.uniform(bounds[j, 0], bounds[j, 1])

        new_pop = np.clip(new_pop, bounds[:, 0], bounds[:, 1])
        
        self.population= new_pop
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        worst_idx = np.argmax(self.fitness)
        self.population[worst_idx] = elite_habitat
        self.fitness[worst_idx] = self.problem.evaluate(elite_habitat)

        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()

        return self.best_solution, self.best_fitness
