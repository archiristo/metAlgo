import numpy as np
import pytest
from metAlgo.framework.operators import (
    tournament_selection,
    roulette_wheel_selection,
    arithmetic_crossover,
    order_crossover_ox,
    gaussian_mutation,
    swap_mutation
)

def test_tournament_selection():
    population = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    fitness = np.array([10.0, 2.0, 5.0])
    selected = tournament_selection(population, fitness, tournament_size=3)
    np.testing.assert_array_equal(selected, np.array([2.0, 2.0]))

def test_roulette_wheel_selection():
    population = np.array([[1.0], [2.0], [3.0]])
    fitness = np.array([1.0, 2.0, 3.0])
    selected = roulette_wheel_selection(population, fitness)
    assert selected.shape == (1,)

def test_arithmetic_crossover():
    p1 = np.array([10.0, 20.0])
    p2 = np.array([0.0, 0.0])
    c1, c2 = arithmetic_crossover(p1, p2, alpha=0.5)
    np.testing.assert_array_equal(c1, np.array([5.0, 10.0]))
    np.testing.assert_array_equal(c2, np.array([5.0, 10.0]))

def test_order_crossover_ox():
    p1 = np.array([0, 1, 2, 3, 4])
    p2 = np.array([4, 3, 2, 1, 0])
    
    c1, c2 = order_crossover_ox(p1, p2)
    
    assert len(c1) == 5
    assert len(c2) == 5
    assert set(c1) == {0, 1, 2, 3, 4}
    assert set(c2) == {0, 1, 2, 3, 4}

def test_gaussian_mutation_bounds():
    chrom = np.array([5.0, -5.0])
    bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])

    for _ in range(20):
        mutated = gaussian_mutation(chrom, bounds, mut_rate=1.0, scale=10.0)
        assert np.all(mutated >= bounds[:, 0])
        assert np.all(mutated <= bounds[:, 1])

def test_swap_mutation():
    chrom = np.array([0, 1, 2, 3, 4])
    mutated = swap_mutation(chrom, mut_rate=1.0)
    
    assert len(mutated) == 5
    assert set(mutated) == {0, 1, 2, 3, 4} 
