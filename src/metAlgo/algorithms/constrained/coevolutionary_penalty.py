import numpy as np

class CoevolutionaryPenalty:
    def __init__(self, num_constraints, *args, **kwargs):
        self.penalty_weights = np.ones(num_constraints)

    def evaluate(self, fitness, constraints_violations):
        return fitness + np.sum(self.penalty_weights * constraints_violations)

    def update_weights(self, population_violations):
        violation_rate = np.mean(population_violations > 0, axis=0)
        self.penalty_weights += 0.1 * (violation_rate - 0.5)