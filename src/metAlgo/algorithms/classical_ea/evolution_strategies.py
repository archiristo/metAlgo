import numpy as np
from metAlgo.framework.base_algorithm import BaseAlgorithm
from metAlgo.framework.base_problem import BaseProblem


class BaseEvolutionStrategy(BaseAlgorithm):
    def __init__(self, problem: BaseProblem, mu=1, lam=1, sigma=0.1, max_iters=1000, *args, **kwargs):
        super().__init__(problem)
        self.mu = mu
        self.lam = lam
        self.sigma = sigma
        self.max_iters = max_iters

    def mutate(self, x):
        return x + self.sigma * np.random.randn(*x.shape)


class OnePlusOneES(BaseEvolutionStrategy):
    def run(self):
        x = self.problem.initial_solution()
        fx = self.problem.evaluate(x)
        for _ in range(self.max_iters):
            y = self.mutate(x)
            fy = self.problem.evaluate(y)
            if fy < fx:
                x, fx = y, fy
        return x, fx


class AdaptiveOnePlusOneES(BaseEvolutionStrategy):

    def run(self):
        x = self.problem.initial_solution()
        fx = self.problem.evaluate(x)
        tau = 0.85
        for _ in range(self.max_iters):
            y = self.mutate(x)
            fy = self.problem.evaluate(y)
            if fy < fx:
                x, fx = y, fy
                self.sigma *= 1.2
            else:
                self.sigma *= tau
        return x, fx


class MuPlusOneES(BaseEvolutionStrategy):
    def run(self):
        population = [self.problem.initial_solution() for _ in range(self.mu)]
        fitness = [self.problem.evaluate(ind) for ind in population]
        for _ in range(self.max_iters):
            offspring = [self.mutate(population[np.argmin(fitness)])]
            offspring_fitness = [self.problem.evaluate(ind) for ind in offspring]
            population += offspring
            fitness += offspring_fitness
            idx = np.argsort(fitness)[:self.mu]
            population = [population[i] for i in idx]
            fitness = [fitness[i] for i in idx]
        return population[0], fitness[0]


class MuPlusLambdaES(BaseEvolutionStrategy):
    def run(self):
        population = [self.problem.initial_solution() for _ in range(self.mu)]
        fitness = [self.problem.evaluate(ind) for ind in population]
        for _ in range(self.max_iters):
            offspring = [self.mutate(population[np.random.randint(self.mu)]) for _ in range(self.lam)]
            offspring_fitness = [self.problem.evaluate(ind) for ind in offspring]
            population += offspring
            fitness += offspring_fitness
            idx = np.argsort(fitness)[:self.mu]
            population = [population[i] for i in idx]
            fitness = [fitness[i] for i in idx]
        return population[0], fitness[0]


class MuCommaLambdaES(BaseEvolutionStrategy):
    def run(self):
        population = [self.problem.initial_solution() for _ in range(self.mu)]
        for _ in range(self.max_iters):
            offspring = [self.mutate(population[np.random.randint(self.mu)]) for _ in range(self.lam)]
            offspring_fitness = [self.problem.evaluate(ind) for ind in offspring]
            idx = np.argsort(offspring_fitness)[:self.mu]
            population = [offspring[i] for i in idx]
        best = min(population, key=lambda ind: self.problem.evaluate(ind))
        return best, self.problem.evaluate(best)


class CMAES(BaseEvolutionStrategy):
    def __init__(self, problem: BaseProblem, mu=None, lam=None, sigma=0.3, max_iters=100, *args, **kwargs):
        super().__init__(problem, mu, lam, sigma, max_iters)
        self.n = self.problem.dimension()
        self.lam = lam if lam else 4 + int(3 * np.log(self.n))
        self.mu = mu if mu else self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs

    def initialize(self):
        if hasattr(self, 'population') and self.population is not None and len(self.population) > 0:
            return

        self.n = self.problem.dimension()
        
        self.mean = self.problem.initial_solution()
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)
        self.C = np.eye(self.n)
        self.invsqrtC = self.B @ np.diag(1 / np.maximum(self.D, 1e-20)) @ self.B.T
        self.eigeneval = 0
        self.chiN = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))
        self.history = []

        self.population = np.array([self.problem.initial_solution() for _ in range(self.lam)])
        
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
    
    def step(self):
        arz = np.random.randn(self.lam, self.n)
        ary = arz @ (self.B @ np.diag(self.D)).T
        arx = self.mean + self.sigma * ary
        fitness = np.array([self.problem.evaluate(x) for x in arx])
        idx = np.argsort(fitness)
        arx, arz = arx[idx], arz[idx]
        mean_old = self.mean
        self.mean = np.dot(self.weights, arx[:self.mu])
        y = np.dot(self.weights, arz[:self.mu])
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff) * (self.invsqrtC @ y)
        hsig = int((np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*(self.eigeneval+1)))/self.chiN) < (1.4+2/(self.n+1)))
        self.pc = (1 - self.cc) * self.pc + hsig*np.sqrt(self.cc*(2-self.cc)*self.mueff)*(self.mean-mean_old)/self.sigma
        artmp = (arx[:self.mu] - mean_old)/self.sigma
       
        self.C = 0.5 * (self.C + self.C.T)
        self.C += np.eye(self.n) * 1e-12  
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(self.D, 1e-20))
        self.sigma *= np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chiN - 1))
        eps = 1e-8
        safe_D = np.maximum(self.D, eps)
        self.invsqrtC = self.B @ np.diag(1.0 / np.sqrt(safe_D)) @ self.B.T
        self.history.append(min(fitness))
        return self.mean, min(fitness)


    def run(self):
        self.initialize()
        best_sol, best_fit = None, float("inf")
        for gen in range(self.max_iters):
            sol, fit = self.step()
            if fit < best_fit:
             best_sol, best_fit = sol, fit
        return best_sol, best_fit

