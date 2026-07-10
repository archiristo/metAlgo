import numpy as np

def fast_non_dominated_sort(fitness_matrix):
    pop_size = fitness_matrix.shape[0]
    domination_count = np.zeros(pop_size) 
    dominated_solutions = [[] for _ in range(pop_size)] 
    fronts = [[]] 

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            if np.all(fitness_matrix[i] <= fitness_matrix[j]) and np.any(fitness_matrix[i] < fitness_matrix[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif np.all(fitness_matrix[j] <= fitness_matrix[i]) and np.any(fitness_matrix[j] < fitness_matrix[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1
    fronts[0] = [i for i in range(pop_size) if domination_count[i] == 0]
    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)
        
    return fronts[:-1]