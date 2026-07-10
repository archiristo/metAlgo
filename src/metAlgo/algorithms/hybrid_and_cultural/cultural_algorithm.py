import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm

class CulturalAlgorithm(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, belief_size=5, *args, **kwargs):
        super().__init__(problem, name="CulturalAlgorithm")
        self.pop_size = pop_size
        self.dim = problem.dimension()
        self.belief_space = {
            "min": np.full(self.dim, -10.0),
            "max": np.full(self.dim, 10.0),
            "best_ever": np.zeros(self.dim)
        }

    def initialize(self):
        self.population = [self.problem.initial_solution() for _ in range(self.pop_size)]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]
    def step(self):
        for i in range(self.pop_size):
            influence = np.random.uniform(self.belief_space["min"], self.belief_space["max"])
            self.population[i] = 0.5 * (self.population[i] + influence)
            self.fitness[i] = self.problem.evaluate(self.population[i])

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.problem.evaluate(self.belief_space["best_ever"]):
            self.belief_space["best_ever"] = np.copy(self.population[best_idx])
            self.belief_space["min"] = np.maximum(self.belief_space["min"], self.belief_space["best_ever"] - 2.0)
            self.belief_space["max"] = np.minimum(self.belief_space["max"], self.belief_space["best_ever"] + 2.0)

        self.best_solution, self.best_fitness = self.population[best_idx], self.fitness[best_idx]
        return self.best_solution, self.best_fitness