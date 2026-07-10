from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
import numpy as np


class BaseProblem(ABC):
    def __init__(
        self,
        dim: int,
        bounds: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
        num_objectives: int = 1,
        num_constraints: int = 0,
        name: str = "GenericProblem"
    ):
        self.dim = dim
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.name = name
        if bounds is not None:
            self.bounds = np.array(bounds, dtype=float)
            if self.bounds.shape != (dim, 2):
                raise ValueError(f"The shape of the bounds must be ({dim}, 2). Received: {self.bounds.shape}")
        else:
            self.bounds = None

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> Union[float, np.ndarray]:
        pass

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(self.num_constraints)

    def is_discrete(self) -> bool:
        return self.bounds is None

    def __repr__(self) -> str:
        return f"<{self.name} | Dim: {self.dim} | Objectives: {self.num_objectives} | Constraints: {self.num_constraints}>"
    
    def initial_solution(self) -> np.ndarray:
        return np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
