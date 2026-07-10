import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm

class UMDA(BaseAlgorithm):
    def __init__(self, problem, pop_size=50, selection_size=20, *args, **kwargs):
        super().__init__(problem, name="UMDA")
        self.pop_size = pop_size
        self.selection_size = selection_size
        self.dim = problem.dimension()
        self.mean = np.zeros(self.dim)
        self.std = np.ones(self.dim) * 10.0 

    def initialize(self):
        self.population = np.random.randn(self.pop_size, self.dim) * self.std + self.mean
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]

    def step(self):
        fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        idx = np.argsort(fitness)
        selected_pop = self.population[idx[:self.selection_size]]
        self.mean = np.mean(selected_pop, axis=0)
        self.std = np.std(selected_pop, axis=0) + 1e-6 
        self.population = np.random.randn(self.pop_size, self.dim) * self.std + self.mean
        best_idx = np.argmin(fitness)
        self.best_solution, self.best_fitness = self.population[best_idx], fitness[best_idx]
        return self.best_solution, self.best_fitness