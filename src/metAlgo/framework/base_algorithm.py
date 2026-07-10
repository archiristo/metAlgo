from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, List
import numpy as np
from metAlgo.framework.base_problem import BaseProblem
from metAlgo.analytics.logger import Logger
class BaseAlgorithm(ABC):

    def __init__(self, problem: BaseProblem, name: str = "GenericAlgorithm"):
        self.problem = problem
        self.name = name
        self.population = np.array([]) 
        self.fitness = np.array([])
        self.best_solution = None
        self.best_fitness = float('inf')
        
        self.logger = Logger(filename=f"{self.name}_log.csv")
        
        self.current_iteration: int = 0
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        self.history: List[float] = []

    def reset(self) -> None:
        self.current_iteration = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        self.logger = Logger(filename=f"{self.name}_log.csv")

    def initialize(self):
        raise NotImplementedError("Subclasses must initialize!")
    
    @abstractmethod
    def step(self) -> Tuple[np.ndarray, float]:
        pass

    def solve(self, max_iterations: int, callback: Optional[Any] = None) -> Tuple[np.ndarray, float]:
        self.reset()
        self.initialize()

        for _ in range(max_iterations):
            self.current_iteration += 1
            
            best_sol, best_fit = self.step()
            self.history.append(best_fit)

            mean_fit = np.mean(getattr(self, 'fitness', [best_fit]))
            
            self.logger.log(
                self.name, 
                self.problem.name, 
                self.current_iteration, 
                self.best_fitness, 
                mean_fit
            )

            if callback is not None:
                callback(self)

        return self.best_solution, self.best_fitness