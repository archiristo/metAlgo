import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm

class Genocop(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, mutation_rate=0.1, *args, **kwargs):
        super().__init__(problem, name="Genocop")
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

    def initialize(self):
        self.population = []
        max_attempts = self.pop_size * 50 
        attempts = 0
        check_func = getattr(self.problem, 'check_constraints', lambda x: True)
        
        while len(self.population) < self.pop_size and attempts < max_attempts:
            ind = self.problem.initial_solution()
            if check_func(ind):
                self.population.append(ind)
            attempts += 1
            
        while len(self.population) < self.pop_size:
            self.population.append(self.problem.initial_solution())
            
        self.population = np.array(self.population)
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]

    def step(self):
        check_func = getattr(self.problem, 'check_constraints', lambda x: True)
        
        for i in range(self.pop_size):
            if np.random.rand() < self.mutation_rate:
                candidate = self.population[i] + np.random.normal(0, 0.1, self.problem.dimension())
                
                if check_func(candidate): 
                    fit = self.problem.evaluate(candidate)
                    if fit < self.fitness[i]:
                        self.population[i] = candidate
                        self.fitness[i] = fit

        best_idx = np.argmin(self.fitness)
        self.best_solution, self.best_fitness = self.population[best_idx], self.fitness[best_idx]
        return self.best_solution, self.best_fitness