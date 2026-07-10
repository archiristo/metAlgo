import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm
from metAlgo.algorithms.multi_objective.pareto_sorting import fast_non_dominated_sort

class NSGAII(BaseAlgorithm):
    def __init__(self, problem, pop_size=50, *args, **kwargs):
        super().__init__(problem, name="NSGA-II")
        self.pop_size = pop_size

    def initialize(self):
        self.population = np.random.uniform(self.problem.lower_bound, self.problem.upper_bound, (self.pop_size, self.problem.dimension()))
        self.fitness = np.array([[self.problem.evaluate(ind)] for ind in self.population])
        self.best_fitness = float(np.min(self.fitness))
    def step(self):
        fronts = fast_non_dominated_sort(self.fitness)
        
        new_pop = []
        while len(new_pop) < self.pop_size:
            idx1, idx2 = np.random.randint(0, self.pop_size, 2)
            if self.fitness[idx1][0] < self.fitness[idx2][0]: 
                parent = self.population[idx1]
            else:
                parent = self.population[idx2]
            
            child = parent + np.random.normal(0, 0.1, self.problem.dimension())
            new_pop.append(np.clip(child, self.problem.lower_bound, self.problem.upper_bound))
        
        self.population = np.array(new_pop)
        self.fitness = np.array([[self.problem.evaluate(ind)] for ind in self.population])
        return self.population[0], self.fitness[0]