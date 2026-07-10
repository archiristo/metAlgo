import numpy as np

class MOOStrategy:

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def vega(fitness_matrix, sub_population_size):
        
        pass

    @staticmethod
    def lexicographic(fitness_matrix, priority_order):
       
        return np.lexsort(fitness_matrix.T[priority_order[::-1]])

    @staticmethod
    def epsilon_constraint(fitness_matrix, epsilons, target_obj_idx):
        
        feasible = np.all(fitness_matrix[:, [i for i in range(fitness_matrix.shape[1]) if i != target_obj_idx]] <= epsilons, axis=1)
        return feasible

    @staticmethod
    def gender_based(population, fitness_matrix):
        pass