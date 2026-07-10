import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
from metAlgo.analytics.benchmark_generator import ShiftedSphereProblem, RastriginProblem, AckleyProblem, RosenbrockProblem
from metAlgo.analytics.stats import run_statistical_test, run_statistical_analysis
from metAlgo.analytics.latex_exporter import generate_ieee_latex_table
from metAlgo.analytics.visualizer import plot_convergence_curves
from metAlgo.framework.factory import AlgorithmFactory
warnings.filterwarnings("ignore", category=RuntimeWarning)
def main():
    
    problems = [
        RastriginProblem(dim=10), 
        AckleyProblem(dim=10), 
        ShiftedSphereProblem(dim=10, shift_value=2.5), 
        RosenbrockProblem(dim=10)
    ]
    
    algo_configs = {
        "RandomMutationHillClimbing": {"mutation_scale": 0.5},
        "GeneticAlgorithm": {"pop_size": 30, "mut_rate": 0.1},
        "ParticleSwarmOptimization": {"pop_size": 30},
        "BiogeographyBasedOptimization": {"pop_size": 30},
        "OppositionalBBO": {"pop_size": 30},
        "DifferentialEvolution": {"pop_size": 30},
        "SimpleEP": {"mu": 30},
        "SimpleGP": {"pop_size": 30, "max_depth": 3, "terminals": ['x0', '1'], "functions": ['add', 'mul']},
        "FireflyAlgorithm": {"pop_size": 30, "alpha": 0.5, "gamma": 1.0},
        "ArtificialBeeColony": {"pop_size": 30, "limit": 20},
        "FishSwarm": {"pop_size": 30, "visual": 1.0, "step": 0.5},
        "AntColony": {"pop_size": 30, "alpha": 1.0, "beta": 2.0, "rho": 0.1},
        "UMDA": {"pop_size": 50, "selection_size": 20},
        "ABCDE": {"pop_size": 30, "limit": 20, "mut_factor": 0.8},
        "CulturalAlgorithm": {},
        "TabuSearch": {"tabu_tenure": 5},
        "Genocop": {"pop_size": 30},
        "SimulatedAnnealing": {"initial_temp": 1000, "cooling_rate": 0.95},
        "MOBBO": {"pop_size": 30},
        "CompactGA": {"pop_size": 30, "learning_rate": 0.1},
        "MultivariateEDA": {"pop_size": 30, "learning_rate": 0.1},
        "CMAES": {"pop_size": 30},
        "MetaEP": {"mu": 30, "lambda_": 100},
        "DiscreteEP": {"mu": 30, "lambda_": 100},
        "NSGAII": {"pop_size": 30}
    }

    for problem in problems:
        print(f"\n{problem.name}")
        results_single_problem = []
        
        for algo_name, params in algo_configs.items():
            try:
                algo_cls = AlgorithmFactory.get_class(algo_name)
                res = run_statistical_test(algo_cls, problem, params)
                if res:
                    results_single_problem.append(res)
                    print(f"{algo_name} has completed.")
            except Exception as e:
                print(f"Error: {algo_name} - {e}")
        
        if len(results_single_problem) >= 3:
                valid_results = [r for r in results_single_problem if np.all(np.isfinite(r["Raw"]))]
                final_clean_results = [r for r in valid_results if np.all(np.isfinite(r["Mean"]))]
                

                if len(final_clean_results) >= 3:
                    seen_algos = set()
                    unique_results = []
                    for r in final_clean_results:
                        if r["Algorithm"] not in seen_algos:
                            unique_results.append(r)
                            seen_algos.add(r["Algorithm"])
                    print(f"\n{problem.name} IEEE Report")
                    print(generate_ieee_latex_table(unique_results, problem.name))
                    final_clean_raw_data = [r["Raw"] for r in unique_results]
                    run_statistical_analysis(final_clean_raw_data)
                    
                    history_data = [{"name": r["Algorithm"], "history": r["History"]} for r in unique_results]
                    plot_convergence_curves(history_data)
                else:
                    print(f"\n--- {problem.name}: No adequate algorithm yielding a valid (finite) result was found.")
        else:
                print(f"\n {problem.name}: There are not enough successful algorithms.")

if __name__ == "__main__":
    main()