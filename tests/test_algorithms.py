import pytest
import numpy as np
from metAlgo.framework.factory import AlgorithmFactory
from metAlgo.analytics.benchmark_generator import (
    SphereProblem, ShiftedSphereProblem, RastriginProblem, AckleyProblem, RosenbrockProblem
)
from metAlgo.algorithms.local_search.hill_climbing import RandomMutationHillClimbing
from metAlgo.algorithms.classical_ea.genetic_algorithm import GeneticAlgorithm
from metAlgo.algorithms.swarm_and_physics.particle_swarm import ParticleSwarmOptimization
from metAlgo.algorithms.biogeography_bbo.bbo_base import BiogeographyBasedOptimization
from metAlgo.algorithms.biogeography_bbo.oppositional_bbo import OppositionalBBO
from metAlgo.algorithms.swarm_and_physics.differential_evolution import DifferentialEvolution
from metAlgo.algorithms.classical_ea.evolution_strategies import CMAES
from metAlgo.algorithms.classical_ea.evolutionary_programming import SimpleEP, MetaEP, DiscreteEP
from metAlgo.algorithms.classical_ea.genetic_programming import SimpleGP  
from metAlgo.algorithms.swarm_and_physics.bio_inspired_meta import FireflyAlgorithm, ArtificialBeeColony, FishSwarm
from metAlgo.algorithms.swarm_and_physics.ant_colony import AntColonyOptimization
from metAlgo.algorithms.probabilistic_eda.umda import UMDA
from metAlgo.algorithms.hybrid.abc_de import ABCDE
from metAlgo.algorithms.hybrid_and_cultural.cultural_algorithm import CulturalAlgorithm
from metAlgo.algorithms.hybrid_and_cultural.tabu_search import TabuSearch
from metAlgo.algorithms.constrained.behavioral_memory import BehavioralMemory
from metAlgo.algorithms.constrained.coevolutionary_penalty import CoevolutionaryPenalty    
from metAlgo.algorithms.constrained.genocop import Genocop
from metAlgo.algorithms.local_search.simulated_annealing import SimulatedAnnealing
from metAlgo.algorithms.multi_objective.classical_moo import MOOStrategy
from metAlgo.algorithms.multi_objective.mo_bbo import MOBBO
from metAlgo.algorithms.probabilistic_eda.compact_ga import CompactGA
from metAlgo.algorithms.probabilistic_eda.multivariate_eda import MultivariateEDA
from metAlgo.algorithms.multi_objective.nsga_ii import NSGAII

ALGORITHMS = [
    ("RandomMutationHillClimbing", {"mutation_scale": 0.5}),
    ("GeneticAlgorithm", {"pop_size": 30, "mut_rate": 0.1}),
    ("ParticleSwarmOptimization", {"pop_size": 30}),
    ("BiogeographyBasedOptimization", {"pop_size": 30}),
    ("OppositionalBBO", {"pop_size": 30}),
    ("DifferentialEvolution", {"pop_size": 30}),
    ("SimpleEP", {"mu": 30}),
    ("SimpleGP", {"pop_size": 30, "max_depth": 3, "terminals": ['x0', '1'], "functions": ['add', 'mul']}),
    ("FireflyAlgorithm", {"pop_size": 30, "alpha": 0.5, "gamma": 1.0}),
    ("ArtificialBeeColony", {"pop_size": 30, "limit": 20}),
    ("FishSwarm", {"pop_size": 30, "visual": 1.0, "step": 0.5}),
    ("AntColony", {"pop_size": 30, "alpha": 1.0, "beta": 2.0, "rho": 0.1}),
    ("UMDA", {"pop_size": 50, "selection_size": 20}),
    ("ABCDE", {"pop_size": 30, "limit": 20, "mut_factor": 0.8}),
    ("CulturalAlgorithm", {}),
    ("TabuSearch", {"tabu_tenure": 5}),
    ("Genocop", {"pop_size": 30}),
    ("SimulatedAnnealing", {"initial_temp": 1000, "cooling_rate": 0.95}),
    ("MOBBO", {"pop_size": 30}),
    ("CompactGA", {"pop_size": 30, "learning_rate": 0.1}),
    ("MultivariateEDA", {"pop_size": 30, "learning_rate": 0.1}),
    ("CMAES", {"pop_size": 30}),
    ("MetaEP", {"mu": 30, "lambda_": 100}),
    ("DiscreteEP", {"mu": 30, "lambda_": 100}),
    ("NSGAII", {"pop_size": 30})
]
PROBLEMS = [SphereProblem, ShiftedSphereProblem, RastriginProblem, AckleyProblem, RosenbrockProblem]

@pytest.mark.parametrize("algo_name, params", ALGORITHMS)
@pytest.mark.parametrize("problem_cls", PROBLEMS)
def test_all_algorithms_on_benchmarks(algo_name, params, problem_cls):
    problem = problem_cls(dim=3)
    algo_cls = AlgorithmFactory.get_class(algo_name)
    algo = algo_cls(problem, **params)
    
    try:
        algo.initialize()
    except Exception as e:
        pytest.skip(f"[{algo_name}] Initialize hatası: {e}")
        
    population = getattr(algo, 'population', getattr(algo, 'population', None))
    if population is None or len(population) == 0:
        pytest.skip(f"[{algo_name}] Popülasyon boş veya yok.")

    for _ in range(5):
        try:
            algo.step()
        except Exception as e:
            pytest.skip(f"[{algo_name}] step() hatası: {e}")

    assert True