import numpy as np
from typing import Tuple
from metAlgo.framework.base_algorithm import BaseAlgorithm
from metAlgo.framework.base_problem import BaseProblem

class RandomMutationHillClimbing(BaseAlgorithm):
    def __init__(self, problem: BaseProblem, mutation_scale: float = 0.1, name: str = "RMHC", *args, **kwargs):
        super().__init__(problem, name)
        self.mutation_scale = mutation_scale
        self.current_solution = None
        self.bounds = self.problem.bounds

    def initialize(self) -> None:
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        
        self.current_solution = np.random.uniform(
            low=lower_bounds, 
            high=upper_bounds, 
            size=self.problem.dim
        )
        
        self.current_sol = self.problem.initial_solution() 
        self.best_solution = self.current_sol
        self.best_fitness = self.problem.evaluate(self.current_sol)
        self.population = np.array([self.current_sol]) 

    def step(self) -> Tuple[np.ndarray, float]:
        noise = np.random.normal(loc=0.0, scale=self.mutation_scale, size=self.problem.dim)
        candidate_solution = self.current_solution + noise
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        candidate_solution = np.clip(candidate_solution, lower_bounds, upper_bounds)
        
        candidate_fitness = self.problem.evaluate(candidate_solution)
        if candidate_fitness < self.best_fitness:
            self.current_solution = candidate_solution
            self.best_solution = candidate_solution.copy()
            self.best_fitness = candidate_fitness
            
        return self.best_solution, self.best_fitness