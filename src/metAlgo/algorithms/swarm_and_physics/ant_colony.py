import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm

class AntColonyOptimization(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, alpha=1.0, beta=2.0, rho=0.1, *args, **kwargs):
        super().__init__(problem, name="AntColony")
        self.pop_size = pop_size
        self.alpha = alpha 
        self.beta = beta  
        self.rho=rho
        self.pheromone = np.ones((problem.dimension(), problem.dimension()))

    def initialize(self):
        self.population = [self.problem.initial_solution() for _ in range(self.pop_size)]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]
    def step(self):
        for i in range(self.pop_size):
            probs = self.pheromone / np.sum(self.pheromone)
            probs = probs.flatten()
            probs /= np.sum(probs)
            move = np.random.choice(len(probs), p=probs.flatten())
            self.population[i] += np.random.normal(0, 0.1, size=self.population[i].shape)
            self.population[i] = np.clip(self.population[i], self.problem.bounds[0][0], self.problem.bounds[0][1])
            self.fitness[i] = self.problem.evaluate(self.population[i])
        self.pheromone *= (1 - self.rho)
        best_idx = np.argmin(self.fitness)
        self.pheromone += 1.0 / (1.0 + self.fitness[best_idx])

        self.best_solution, self.best_fitness = self.population[best_idx], self.fitness[best_idx]
        return self.best_solution, self.best_fitness