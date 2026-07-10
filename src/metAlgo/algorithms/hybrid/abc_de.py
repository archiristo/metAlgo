import numpy as np
from metAlgo.algorithms.swarm_and_physics.differential_evolution import DifferentialEvolution
from metAlgo.algorithms.swarm_and_physics.bio_inspired_meta import ArtificialBeeColony

class ABCDE(ArtificialBeeColony):
    def __init__(self, problem, pop_size=30, limit=20, mut_factor=0.8, *args, **kwargs):
        super().__init__(problem, pop_size, limit)
        self.mut_factor = mut_factor 

    def step(self):
        super().step() 
        for i in range(self.pop_size):
            if self.counter[i] > (self.limit / 2): 
                a, b = np.random.choice(self.pop_size, 2, replace=False)
                mutated = self.population[i] + self.mut_factor * (self.population[a] - self.population[b])
            
                if self.problem.evaluate(mutated) < self.fitness[i]:
                    self.population[i] = mutated
                    self.fitness[i] = self.problem.evaluate(mutated)
                    self.counter[i] = 0
        
        return self.best_solution, self.best_fitness