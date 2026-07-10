import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from metAlgo.framework.base_problem import BaseProblem
from metAlgo.algorithms.classical_ea.genetic_algorithm import GeneticAlgorithm
from metAlgo.framework.operators import tournament_selection, order_crossover_ox, swap_mutation

class TSPProblem(BaseProblem):
    def __init__(self, num_cities: int = 30, seed: int = 42):
        super().__init__(dim=num_cities, bounds=None, name=f"TSP-{num_cities}")
    
        np.random.seed(seed)
        self.city_coords = np.random.uniform(0, 100, (num_cities, 2))
        
        self.distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                self.distance_matrix[i, j] = np.linalg.norm(self.city_coords[i] - self.city_coords[j])

    def evaluate(self, x: np.ndarray) -> float:
        total_distance = 0.0
        num_cities = self.dim
        
        for i in range(num_cities):
            current_city = x[i]
            next_city = x[(i + 1) % num_cities]
            total_distance += self.distance_matrix[current_city, next_city]
            
        return float(total_distance)

def run_animated_tsp():
    num_cities = 30
    pop_size = 100
    max_generations = 200


    tsp_prob = TSPProblem(num_cities=num_cities)
    
    ga = GeneticAlgorithm(
        problem=tsp_prob,
        pop_size=pop_size,
        mut_rate=0.05,
        elitism_count=5,
        selection_func=tournament_selection,
        crossover_func=order_crossover_ox,
        mutation_func=swap_mutation
    )

    ga.initialize()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.set_title("Evrimsel Yol Haritası (Canlı)")
    coords = tsp_prob.city_coords
    ax1.scatter(coords[:, 0], coords[:, 1], color='red', s=50, zorder=5)
    for i, (x, y) in enumerate(coords):
        ax1.text(x+1, y+1, str(i), fontsize=9, color='blue')
    line, = ax1.plot([], [], color='green', linewidth=2, zorder=1)

    ax2.set_title("Yakınsama Eğrisi (Mesafe / Jenerasyon)")
    ax2.set_xlabel("Jenerasyon")
    ax2.set_ylabel("En Kısa Mesafe")
    convergence_line, = ax2.plot([], [], color='red', linewidth=2)
    ax2.set_xlim(0, max_generations)
    
    initial_best = ga.best_fitness
    ax2.set_ylim(initial_best * 0.3, initial_best * 1.1)
    ax2.grid(True)
    info_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update(frame):
        if ga.current_iteration < max_generations:
            ga.step()
            ga.current_iteration += 1
            ga.history.append(ga.best_fitness)

        best_tour = ga.best_solution
        ordered_coords = coords[best_tour]
        ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])
        
        line.set_data(ordered_coords[:, 0], ordered_coords[:, 1])
        
        convergence_line.set_data(range(len(ga.history)), ga.history)
        
        info_text.set_text(f"Jenerasyon: {ga.current_iteration}\nEn Kısa Mesafe: {ga.best_fitness:.2f}")
        
        return line, convergence_line, info_text

    ani = FuncAnimation(fig, update, frames=max_generations, interval=50, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_animated_tsp()
