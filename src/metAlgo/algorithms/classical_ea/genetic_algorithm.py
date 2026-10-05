import numpy as np
from typing import Tuple, Callable, Optional
from metAlgo.framework.base_problem import BaseProblem
from metAlgo.framework.base_algorithm import BaseAlgorithm
from metAlgo.framework.operators import tournament_selection, arithmetic_crossover, gaussian_mutation

class GeneticAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        problem: BaseProblem,
        pop_size: int = 50,
        mut_rate: float = 0.1,
        elitism_count: int = 2,
        selection_func: Callable = tournament_selection,
        crossover_func: Callable = arithmetic_crossover,
        mutation_func: Callable = gaussian_mutation, *args, **kwargs
    ):
        super().__init__(problem, name="GeneticAlgorithm")
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.elitism_count = elitism_count
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.pop: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None

    def initialize(self) -> None:
        if self.problem.is_discrete():
            self.population= np.array([np.random.permutation(self.problem.dim) for _ in range(self.pop_size)])
        else:
            bounds = self.problem.bounds
            self.population= np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.problem.dim))
        
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        min_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[min_idx]
        self.best_solution = self.population[min_idx].copy()

    def step(self) -> Tuple[np.ndarray, float]:
        new_pop = []
        elite_indices = np.argsort(self.fitness)[:self.elitism_count]
        for idx in elite_indices:
            new_pop.append(self.population[idx].copy())
        current_idx = 0
        if self.elitism_count > 0:
            elite_indices = np.argsort(self.fitness)[:self.elitism_count]
            for idx in elite_indices:
                new_pop[current_idx] = self.population[idx]
                current_idx += 1
        while len(new_pop) < self.pop_size:
            parent1 = self.selection_func(self.population, self.fitness)
            parent2 = self.selection_func(self.population, self.fitness)
            
            child1, child2 = self.crossover_func(parent1, parent2)
            
            if self.problem.is_discrete():
                child1 = self.mutation_func(child1, mut_rate=self.mut_rate)
                child2 = self.mutation_func(child2, mut_rate=self.mut_rate)
            else:
                child1 = self.mutation_func(child1, self.problem.bounds, mut_rate=self.mut_rate)
                child2 = self.mutation_func(child2, self.problem.bounds, mut_rate=self.mut_rate)
            
            new_pop.append(child1)
            if len(new_pop) < self.pop_size:
                new_pop.append(child2)
        self.population= np.array(new_pop)
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()
            
        return self.best_solution, self.best_fitness
