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

class AlgorithmFactory:
    _algorithms = {
        "RandomMutationHillClimbing": RandomMutationHillClimbing,
        "GeneticAlgorithm": GeneticAlgorithm,
        "ParticleSwarmOptimization": ParticleSwarmOptimization,
        "BiogeographyBasedOptimization": BiogeographyBasedOptimization,
        "OppositionalBBO": OppositionalBBO,
        "DifferentialEvolution": DifferentialEvolution,
        "SimpleEP": SimpleEP,
        "SimpleGP": SimpleGP,
        "FireflyAlgorithm": FireflyAlgorithm,
        "ArtificialBeeColony": ArtificialBeeColony,
        "FishSwarm": FishSwarm,
        "AntColony": AntColonyOptimization,
        "UMDA": UMDA,
        "ABCDE": ABCDE,
        "NSGAII": NSGAII,
        "TabuSearch": TabuSearch,
        "CulturalAlgorithm": CulturalAlgorithm,
        "BehavioralMemoryAlgorithm": BehavioralMemory,
        "CoevolutionaryPenalty": CoevolutionaryPenalty,
        "Genocop": Genocop,
        "SimulatedAnnealing": SimulatedAnnealing,
        "ClassicalMOO": MOOStrategy,
        "MOBBO": MOBBO,
        "CompactGA": CompactGA,
        "MultivariateEDA": MultivariateEDA,
        "CMAES": CMAES,
        "MetaEP": MetaEP,
        "DiscreteEP": DiscreteEP
    }

    @classmethod
    def get_class(cls, name):
        if name not in cls._algorithms:
            raise ValueError(f"Algorithm '{name}' not found!")
        return cls._algorithms[name]
    
    @classmethod
    def get_all_names(cls):
        return cls._algorithms.keys()
    
    