import numpy as np

def tournament_selection(population: np.ndarray, fitness: np.ndarray, tournament_size: int = 3) -> np.ndarray:
    pop_size = len(population)
    chosen_indices = np.random.choice(pop_size, tournament_size, replace=False)
    best_idx = chosen_indices[np.argmin(fitness[chosen_indices])]
    
    return population[best_idx].copy()

def roulette_wheel_selection(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    shifted_fitness = np.max(fitness) - fitness + 1e-6
    probabilities = shifted_fitness / np.sum(shifted_fitness)
    chosen_idx = np.random.choice(len(population), p=probabilities)
    return population[chosen_idx].copy()
