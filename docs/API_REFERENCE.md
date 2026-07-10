# API Reference: metAlgo v1.0

This reference details the core components, class structures, and analysis tools of the `metAlgo` library.

## 1. Core Framework

All algorithms and problems must inherit from the following core classes to ensure consistency.

### `BaseAlgorithm`
This is the main class from which all meta-heuristic algorithms are derived.
* `__init__(problem, name)`: Connects the algorithm to a problem object.
* `initialize()`: Creates the population or initial state.
* `step()`: Performs a single iteration of the algorithm.
* `solve(max_iterations)`: Runs the main loop of the algorithm and returns the best solution.

### `BaseProblem`
The class where optimization problems are defined.
* `evaluate(solution)`: Calculates the fitness value of a given solution.
* `lower_bound`, `upper_bound`: Defines the boundaries of the problem space.

---

## 2. Statistical Analysis Tools

The `analytics` module is used to report experimental results and perform significance tests.

### `run_statistical_test(algorithm_class, problem, algo_params, num_runs=30)`
Runs the algorithm in a specified number of independent runs.
* **Parameters:**
* `algorithm_class`: The algorithm class to be run.
* `problem`: An example of the problem to be solved.
* `algo_params`: Algorithm hyperparameter dictionary.
* **Return:** Statistical summary (Mean, Std, Best, Worst, Raw, History).

### `run_statistical_analysis(results_list)`
Performs the **Friedman Test** on the given list of results.
* **Input:** `results_list` (list containing the `Raw` results of each algorithm).
* **Output:** p-value and statistical significance interpretation.

---

## 3. Reporting and Visualization

### `generate_ieee_latex_table(results, problem_name)`
Converts the results into a `LaTeX` table that can be directly added to academic publications.
* **Format:** IEEE standard, `Mean`, `Std Dev`, `Best`, `Worst` columns.

### `plot_convergence_curves(history_data)`
Plots the convergence curves of the algorithms.
* **Input:** A list of the structure `{"name": str, "history": np.array}`.

---

## 4. Algorithm Parameters (Example)

The most frequently needed hyperparameter definitions for users:

| Algorithm | Parameter | Type | Description |
| :--- | :--- | :--- | :--- |
| `ParticleSwarm` | `pop_size` | int | Population size |
| `SimpleGP` | `max_depth` | int | Maximum tree depth |
| `SimulatedAnnealing` | `cooling_rate` | float | Cooling coefficient |

---
*For detailed technical questions, continue reviewing the `metAlgo` documentation.*