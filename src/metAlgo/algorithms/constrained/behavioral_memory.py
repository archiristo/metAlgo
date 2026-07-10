import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm 

class BehavioralMemory(BaseAlgorithm):  
    def __init__(self, problem, memory_size=100, pop_size=30, *args, **kwargs):
        super().__init__(problem, name="BehavioralMemory") 
        self.memory_size = memory_size
        self.trajectory_buffer = []
        self.pop_size = pop_size

    def initialize(self):
        self.population = np.random.uniform(self.problem.lower_bound, self.problem.upper_bound, (30, self.problem.dimension()))
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.best_solution = self.population[np.argmin(self.fitness)]
        self.best_fitness = min(self.fitness)
    
    def step(self):
        bias = self.get_bias()
       
        for i in range(self.pop_size):

            alpha = 0.8  
            
            mutation = np.random.normal(0, 0.1, self.problem.dimension())
            new_ind = self.population[i] + alpha * bias + mutation
            new_ind = np.clip(new_ind, self.problem.lower_bound, self.problem.upper_bound)
            
            new_fit = self.problem.evaluate(new_ind)
            
            if new_fit < self.fitness[i]:
                self.record(new_ind, new_fit)
                
                self.population[i] = new_ind
                self.fitness[i] = new_fit

        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
        
        return self.best_solution, self.best_fitness
    
    def get_bias(self):
        if not self.trajectory_buffer:
            return np.zeros(self.problem.dimension())
        return np.mean(self.trajectory_buffer, axis=0)