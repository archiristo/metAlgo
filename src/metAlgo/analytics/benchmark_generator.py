import numpy as np
from metAlgo.framework.base_problem import BaseProblem

class SphereProblem(BaseProblem):
    def __init__(self, dim: int = 10):
        self.bounds = np.array([(-5.12, 5.12) for _ in range(dim)])
        self.lower_bound = self.bounds[:, 0]
        self.upper_bound = self.bounds[:, 1]
        super().__init__(dim=dim, bounds=self.bounds, name="Sphere")

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))
    
    def dimension(self) -> int:
        return self.dim
    
    def initial_solution(self) -> np.ndarray:
        return np.array([np.random.uniform(low, high) for (low, high) in self.bounds])

class ShiftedSphereProblem(BaseProblem):
    def __init__(self, dim: int = 10, shift_value: float = 2.5):
        self.bounds = np.array([(-5.12, 5.12) for _ in range(dim)])
        self.lower_bound = self.bounds[:, 0]
        self.upper_bound = self.bounds[:, 1]
        super().__init__(dim=dim, bounds=self.bounds, name=f"ShiftedSphere(shift={shift_value})")
        
        self.shift_value = shift_value

    def evaluate(self, x: np.ndarray) -> float:
        x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        if np.any(np.isnan(x)): return 1e10 
        return float(np.sum((x - self.shift_value) ** 2))

    def dimension(self) -> int:
        return self.dim
    
    def initial_solution(self) -> np.ndarray:
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

class RastriginProblem(BaseProblem):
    def __init__(self, dim: int = 10):
        self.bounds = np.array([(-5.12, 5.12) for _ in range(dim)])
        self.lower_bound = self.bounds[:, 0]
        self.upper_bound = self.bounds[:, 1]
        super().__init__(dim=dim, bounds=self.bounds, name="Rastrigin")

    def evaluate(self, x: np.ndarray) -> float:
        x = np.clip(x, self.lower_bound, self.upper_bound) 
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    def dimension(self) -> int:
        return self.dim
    
    def check_constraints(self, x): return True

    def initial_solution(self) -> np.ndarray:
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

class AckleyProblem(BaseProblem):
    def __init__(self, dim: int = 10):
        self.bounds = np.array([(-32.768, 32.768) for _ in range(dim)])
        self.lower_bound = self.bounds[:, 0]
        self.upper_bound = self.bounds[:, 1]
        super().__init__(dim=dim, bounds=self.bounds, name="Ackley")

    def evaluate(self, x: np.ndarray) -> float:
        return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) \
               - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e
    def dimension(self) -> int:
        return self.dim
    
    def initial_solution(self) -> np.ndarray:
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

class RosenbrockProblem(BaseProblem):
    def __init__(self, dim: int = 10):
        self.bounds = np.array([(-5, 10) for _ in range(dim)])
        self.lower_bound = self.bounds[:, 0]
        self.upper_bound = self.bounds[:, 1]
        super().__init__(dim=dim, bounds=self.bounds, name="Rosenbrock")

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def dimension(self) -> int:
        return self.dim
    
    def initial_solution(self) -> np.ndarray:
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)