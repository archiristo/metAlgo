import numpy as np
from typing import Dict, Any, Type, List
from metAlgo.framework.base_problem import BaseProblem
from scipy.stats import wilcoxon, friedmanchisquare

def run_statistical_test(algorithm_class, problem, algo_params, num_runs=30, max_iterations=100):
    from metAlgo.framework.base_algorithm import BaseAlgorithm
    fitness_results = []
    histories = []
    
    for seed in range(num_runs):
        np.random.seed(seed)
        algo = algorithm_class(problem=problem, **algo_params)
        
        res = algo.solve(max_iterations=max_iterations)
        final_best_fit = res[1] 
        if isinstance(final_best_fit, (list, np.ndarray)):
            final_best_fit = float(np.min(final_best_fit))
        else:
            final_best_fit = float(final_best_fit)
        if np.isfinite(final_best_fit):
            fitness_results.append(final_best_fit)
            
        hist = np.array(algo.history)
        if len(hist) == 0:
            hist = np.zeros(max_iterations)
        elif len(hist) < max_iterations:
            hist = np.pad(hist, (0, max_iterations - len(hist)), mode='constant', constant_values=hist[-1])
        elif len(hist) > max_iterations:
            hist = hist[:max_iterations]
            
        histories.append(hist)
    return {
        "Algorithm": algo.name,
        "Mean": np.mean(fitness_results) if fitness_results else float('inf'),
        "Std": np.std(fitness_results) if fitness_results else 0.0,
        "Best": np.min(fitness_results) if fitness_results else float('inf'),
        "Worst": np.max(fitness_results) if fitness_results else float('inf'),
        "Raw": fitness_results,
        "History": np.mean(histories, axis=0)
    }

def run_statistical_analysis(results_list):
    valid_data = [r for r in results_list if len(r) > 0 and np.all(np.isfinite(r))]
    
    if len(valid_data) < 3:
        print("\n less than 3 data")
        return None
    
    min_len = min(len(r) for r in valid_data)
    standardized_data = [r[:min_len] for r in valid_data]
    
    stat, p_value = friedmanchisquare(*standardized_data)
    
    print(f"\nFriedman Analysis")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: There is a statistically significant difference between the algorithms.")
    else:
        print("Conclusion: The difference between the algorithms could be due to chance.")
        
    return p_value

def compare_pair(data1, data2):
    stat, p_value = wilcoxon(data1, data2)
    return p_value