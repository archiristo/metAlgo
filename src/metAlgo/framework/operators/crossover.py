import numpy as np
from typing import Tuple

def arithmetic_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

def order_crossover_ox(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    size = len(parent1)
    idx1, idx2 = sorted(np.random.choice(size, 2, replace=False))
    
    def _create_child(p1, p2):
        child = np.full(size, -1, dtype=int)
        child[idx1:idx2] = p1[idx1:idx2]
        p2_idx = idx2
        child_idx = idx2
        
        while -1 in child:
            curr_gene = p2[p2_idx % size]
            if curr_gene not in child:
                child[child_idx % size] = curr_gene
                child_idx += 1
            p2_idx += 1
        return child

    child1 = _create_child(parent1, parent2)
    child2 = _create_child(parent2, parent1)
    return child1, child2
