import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm

class MultivariateEDA(BaseAlgorithm):
    def __init__(self, problem, pop_size=50, selection_size=20, dim = 10, *args, **kwargs):
        super().__init__(problem, name="MultivariateEDA")
        self.pop_size = pop_size
        self.selection_size = selection_size
        self.dim = dim
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim) 

    def initialize(self):
        dim = self.problem.dimension()
        self.mean = np.zeros(dim) 
        self.cov = np.eye(dim)  
        self.population = np.random.multivariate_normal(self.mean, self.cov, self.pop_size)
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]

    def step(self):
        fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        idx = np.argsort(fitness)
        selected_pop = self.population[idx[:self.selection_size]]
        self.mean = np.mean(selected_pop, axis=0)
        self.cov = np.cov(selected_pop, rowvar=False)
        self.cov += np.eye(self.problem.dimension()) * 1e-6
        self.population = np.random.multivariate_normal(self.mean, self.cov, self.pop_size)
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        best_idx = np.argmin(self.fitness)
        self.best_solution, self.best_fitness = self.population[best_idx], self.fitness[best_idx]
        return self.best_solution, self.best_fitness