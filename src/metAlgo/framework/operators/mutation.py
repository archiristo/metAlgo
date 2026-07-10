import numpy as np

def gaussian_mutation(chrom: np.ndarray, bounds: np.ndarray, mut_rate: float = 0.1, scale: float = 0.1) -> np.ndarray:
    mutated = chrom.copy()
    mutation_mask = np.random.rand(len(mutated)) < mut_rate
    noise = np.random.normal(0, scale, size=len(mutated))
    mutated[mutation_mask] += noise[mutation_mask]
    mutated = np.clip(mutated, bounds[:, 0], bounds[:, 1])
    
    return mutated

def swap_mutation(chrom: np.ndarray, mut_rate: float = 0.1) -> np.ndarray:
    mutated = chrom.copy()
    if np.random.rand() < mut_rate:
        idx1, idx2 = np.random.choice(len(mutated), 2, replace=False)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated
