import numpy as np
import pytest
from metAlgo.analytics.benchmark_generator import (
    SphereProblem, RastriginProblem, AckleyProblem, RosenbrockProblem, ShiftedSphereProblem
)
ALL_PROBLEMS = [SphereProblem, ShiftedSphereProblem, RastriginProblem, AckleyProblem, RosenbrockProblem]

@pytest.mark.parametrize("problem_cls", ALL_PROBLEMS)
def test_problem_initialization(problem_cls):
    dim = 5
    prob = problem_cls(dim=dim)
    assert prob.dim == dim
    assert prob.num_objectives == 1
    assert prob.bounds.shape == (dim, 2)

@pytest.mark.parametrize("problem_cls", ALL_PROBLEMS)
def test_problem_evaluation_at_origin(problem_cls):
    prob = problem_cls(dim=3)
    origin = np.zeros(3)
    fitness = prob.evaluate(origin)
    if problem_cls in [SphereProblem, RastriginProblem]:
        assert fitness == 0.0
    else:
        assert fitness >= 0.0

@pytest.mark.parametrize("problem_cls", ALL_PROBLEMS)
def test_problem_bounds(problem_cls):
    dim = 3
    prob = problem_cls(dim=dim)
    point = np.random.uniform(prob.bounds[:, 0], prob.bounds[:, 1], dim)

    try:
        fitness = prob.evaluate(point)
        assert np.isfinite(fitness)
    except Exception as e:
        pytest.fail(f"{problem_cls.__name__} değerlendirme sırasında hata verdi: {e}")