
import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm
from metAlgo.framework.base_problem import BaseProblem


class BaseEvolutionaryProgramming(BaseAlgorithm):
   
    def __init__(self, problem: BaseProblem, mu=30, lam=None, sigma=0.1, max_iters=1000, *args, **kwargs):
        super().__init__(problem)
        self.mu = mu
        self.lam = lam if lam else mu
        self.sigma = sigma
        self.max_iters = max_iters

    def mutate(self, x):
        return x + self.sigma * np.random.randn(*x.shape)

    def stochastic_tournament(self, population, fitness, k=10):

        new_pop = []
        pop_size = len(population)
        
        actual_k = min(k, pop_size - 1)
        
        for i in range(pop_size):
            opponents = np.random.choice(pop_size, actual_k, replace=False)
            wins = sum(fitness[i] < fitness[j] for j in opponents)
            
            if wins > (actual_k / 2):
                new_pop.append(population[i])
        return new_pop

class SimpleEP(BaseEvolutionaryProgramming): 
    def __init__(self, problem, mu=30, max_iters=1000, sigma=0.1, *args, **kwargs):
        super().__init__(problem, mu=mu, max_iters=max_iters, sigma=sigma)
        self.population = None 

    def initialize(self):
        self.population = [self.problem.initial_solution() for _ in range(self.mu)]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]
        self.history = []

    def step(self):
     
        offspring = [self.mutate(ind) for ind in self.population]
        offspring_fitness = [self.problem.evaluate(ind) for ind in offspring]
        
        combined = self.population+ offspring
        combined_fitness = self.fitness + offspring_fitness
        
        survivors = self.stochastic_tournament(combined, combined_fitness)
        
        if len(survivors) == 0:
            best_idx = np.argmin(combined_fitness)
            self.population = [combined[best_idx]]
            self.fitness = [combined_fitness[best_idx]]
        else:
            self.population = survivors
            self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
        
        self.history.append(self.best_fitness)
        return self.best_solution, self.best_fitness
    
class MetaEP(BaseEvolutionaryProgramming):
   
    def initialize(self):
        self.population= [self.problem.initial_solution() for _ in range(self.mu)]
        self.sigmas = [self.sigma for _ in range(self.mu)]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.tau = 1 / np.sqrt(2 * np.sqrt(self.problem.dimension()))
        self.tau_prime = 1 / np.sqrt(2 * self.problem.dimension())
        self.history = []
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]

    def step(self):
      
        offspring = [self.mutate(ind) for ind in self.population]
        offspring_fitness = [self.problem.evaluate(ind) for ind in offspring]
        
       
        combined = self.population+ offspring
        combined_fitness = self.fitness + offspring_fitness
        
        survivors = self.stochastic_tournament(combined, combined_fitness)
        
        if len(survivors) == 0:
            best_idx = np.argmin(combined_fitness)
            self.population= [combined[best_idx]]
            self.fitness = [combined_fitness[best_idx]]
        else:
            self.population= survivors
            self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
        
        self.history.append(self.best_fitness)
        return self.best_solution, self.best_fitness

class DiscreteEP(BaseEvolutionaryProgramming):
    
    def __init__(self, problem, mu=30, mutation_rate=0.1, max_iters=1000, *args, **kwargs):
        super().__init__(problem, mu=mu, max_iters=max_iters)
        self.mutation_rate = mutation_rate

    def initialize(self):
        self.population= [self.problem.initial_solution() for _ in range(self.mu)]
        self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        self.history = []
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]

    def step(self):
        offspring = [self.mutate(ind) for ind in self.population]
        offspring_fitness = [self.problem.evaluate(ind) for ind in offspring]
    
        combined = self.population+ offspring
        combined_fitness = self.fitness + offspring_fitness
        
        survivors = self.stochastic_tournament(combined, combined_fitness)
        
        if len(survivors) == 0:
            best_idx = np.argmin(combined_fitness)
            self.population= [combined[best_idx]]
            self.fitness = [combined_fitness[best_idx]]
        else:
            self.population= survivors
            self.fitness = [self.problem.evaluate(ind) for ind in self.population]
        
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
        
        self.history.append(self.best_fitness)
        return self.best_solution, self.best_fitness