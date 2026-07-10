import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm

class FireflyAlgorithm(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, alpha=0.5, beta_min=0.2, gamma=1.0, *args, **kwargs):
        super().__init__(problem, name="FireflyAlgorithm")
        self.pop_size = pop_size
        self.alpha = alpha     
        self.beta_min = beta_min 
        self.gamma = gamma     
        self.population = None
        self.fitness = None

    def initialize(self):
        self.population = [self.problem.initial_solution() for _ in range(self.pop_size)]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]

    def step(self):
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if self.fitness[j] < self.fitness[i]:
                    dist = np.linalg.norm(self.population[i] - self.population[j])
                    beta = np.exp(-self.gamma * dist**2)
                    self.population[i] += beta * (self.population[j] - self.population[i]) + \
                                          self.alpha * (np.random.rand(len(self.population[i])) - 0.5)
                    
                    self.population[i] = np.clip(self.population[i], self.problem.bounds[0][0], self.problem.bounds[0][1])
                    self.fitness[i] = self.problem.evaluate(self.population[i])

        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
        return self.best_solution, self.best_fitness
    
class ArtificialBeeColony(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, limit=20, *args, **kwargs):
        super().__init__(problem, name="ABC")
        self.pop_size = pop_size
        self.limit = limit 
        self.counter = np.zeros(pop_size)

    def initialize(self):
        self.population = [self.problem.initial_solution() for _ in range(self.pop_size)]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]
    def step(self):
        for i in range(self.pop_size):
            k = np.random.randint(self.pop_size)
            phi = np.random.uniform(-1, 1, self.problem.dimension())
            candidate = self.population[i] + phi * (self.population[i] - self.population[k])
            
            f_cand = self.problem.evaluate(candidate)
            if f_cand < self.fitness[i]:
                self.population[i], self.fitness[i] = candidate, f_cand
                self.counter[i] = 0
            else:
                self.counter[i] += 1
        for i in range(self.pop_size):
            if self.counter[i] > self.limit:
                self.population[i] = self.problem.initial_solution()
                self.fitness[i] = self.problem.evaluate(self.population[i])
                self.counter[i] = 0
        
        best_idx = np.argmin(self.fitness)
        self.best_solution, self.best_fitness = self.population[best_idx], self.fitness[best_idx]
        return self.best_solution, self.best_fitness
    
class FishSwarm(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, visual=1.0, step=0.5, *args, **kwargs):
        super().__init__(problem, name="FishSwarm")
        self.pop_size = pop_size
        self.visual = visual
        self.step_size = step

    def initialize(self):
        self.population = [self.problem.initial_solution() for _ in range(self.pop_size)]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]
    def step(self):
        for i in range(self.pop_size):
            best_idx = np.argmin(self.fitness)
            direction = (self.population[best_idx] - self.population[i])
            dist = np.linalg.norm(direction)
            
            if dist > 0:
                self.population[i] += (direction / dist) * self.step_size * np.random.rand()
            
            self.fitness[i] = self.problem.evaluate(self.population[i])

        best_idx = np.argmin(self.fitness)
        self.best_solution, self.best_fitness = self.population[best_idx], self.fitness[best_idx]
        return self.best_solution, self.best_fitness