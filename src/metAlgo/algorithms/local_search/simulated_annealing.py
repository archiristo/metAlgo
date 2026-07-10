import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm

class SimulatedAnnealing(BaseAlgorithm):
    def __init__(self, problem, temp=1000, cooling=0.99, pop_size=30, dim = 2, *args, **kwargs):
        super().__init__(problem, name="SimulatedAnnealing")
        self.temp = temp
        self.cooling = cooling
        self.current_sol = problem.initial_solution()
        self.pop_size = pop_size
        self.dim = dim

    def step(self):
        candidate = self.current_sol + np.random.randn(*self.current_sol.shape)
        delta = self.problem.evaluate(candidate) - self.problem.evaluate(self.current_sol)
        
       
        if delta < 0 or np.random.rand() < np.exp(-delta / self.temp):
            self.current_sol = candidate
            
        self.temp *= self.cooling 
        
        self.best_solution, self.best_fitness = self.current_sol, self.problem.evaluate(self.current_sol)
        return self.best_solution, self.best_fitness
    
    def initialize(self):
     
        bounds = self.problem.bounds
        self.population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.problem.dimension()))
        
       
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
    
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]