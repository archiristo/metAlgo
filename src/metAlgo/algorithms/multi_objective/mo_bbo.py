import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm
from metAlgo.algorithms.multi_objective.pareto_sorting import fast_non_dominated_sort

class MOBBO(BaseAlgorithm):
    def __init__(self, problem, pop_size=50, *args, **kwargs):
        super().__init__(problem, name="MO-BBO")
        self.pop_size = pop_size
        self.population = None 
        self.fitness = None

    def initialize(self):
        self.population = np.array([self.problem.initial_solution() for _ in range(self.pop_size)])
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]
    def step(self):
        check_func = getattr(self.problem, 'check_constraints', lambda x: True)
        
        fronts = fast_non_dominated_sort(self.fitness)
        
        for i in range(self.pop_size):
            if np.random.rand() < 0.5: 
                source = np.random.randint(self.pop_size)
                
                if self.is_better(source, i):
                    candidate = self.population[source] + np.random.normal(0, 0.01, self.problem.dimension())
                    
                    if check_func(candidate):
                        self.population[i] = candidate

        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        if self.population is not None and len(self.population) > 0:
            return self.population[0], self.fitness[0]
        return None, None
    
    def is_better(self, i, j):
        return np.all(self.fitness[i] <= self.fitness[j]) and np.any(self.fitness[i] < self.fitness[j])