from metAlgo.framework.base_algorithm import BaseAlgorithm
import numpy as np

class CompactGA(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, learning_rate=0.1, *args, **kwargs):
        super().__init__(problem, name="CompactGA")
        self.pop_size = pop_size
        self.lr = learning_rate
        self.prob_vector = np.ones(problem.dimension()) * 0.5 

    def step(self):
        ind1 = np.random.rand(self.prob_vector.shape[0]) < self.prob_vector
        ind2 = np.random.rand(self.prob_vector.shape[0]) < self.prob_vector
        if self.problem.evaluate(ind1) < self.problem.evaluate(ind2):
            self.prob_vector += self.lr * (ind1 - self.prob_vector)
        else:
            self.prob_vector += self.lr * (ind2 - self.prob_vector)
        return self.best_solution, self.best_fitness
    
    def initialize(self):
        bounds = self.problem.bounds
        self.population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.problem.dimension()))
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]