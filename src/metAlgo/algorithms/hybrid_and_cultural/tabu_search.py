import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm

class TabuSearch(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, tabu_size=10, *args, **kwargs):
        super().__init__(problem, name="TabuSearch")
        self.pop_size = pop_size
        self.tabu_size = tabu_size
        self.tabu_list = [] 

    def initialize(self):
        self.current_sol = self.problem.initial_solution()
        self.best_solution = np.copy(self.current_sol)
        self.best_fitness = self.problem.evaluate(self.current_sol)
        self.population = np.array([self.current_sol])

    def step(self):
        candidates = [self.current_sol + np.random.randn(*self.current_sol.shape) * 0.1 
                      for _ in range(self.pop_size)]
        best_cand = None
        best_cand_fit = float('inf')
        
        for cand in candidates:
            if not any(np.allclose(cand, t, atol=1e-3) for t in self.tabu_list):
                fit = self.problem.evaluate(cand)
                if fit < best_cand_fit:
                    best_cand, best_cand_fit = cand, fit
        
        if best_cand is not None:
            self.current_sol = best_cand
            self.tabu_list.append(best_cand)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)
            
            if best_cand_fit < self.best_fitness:
                self.best_solution, self.best_fitness = best_cand, best_cand_fit
                
        return self.best_solution, self.best_fitness