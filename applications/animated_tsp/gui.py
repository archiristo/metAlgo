import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from main import TSPProblem
from metAlgo.algorithms.classical_ea.genetic_algorithm import GeneticAlgorithm
from metAlgo.framework.operators import tournament_selection, order_crossover_ox, swap_mutation

class TSPGui:
    def __init__(self, root):
        self.root = root
        self.root.title("metAlgo - Animasyonlu TSP Çözücü Vitrini")
        self.root.geometry("1200x700")
    
        self.is_running = False
        self.max_generations = 300
        self.ani_loop = None

        self.setup_ui()
        self.reset_simulation()

    def setup_ui(self):
        self.left_panel = ttk.Frame(self.root, padding=20, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        self.right_panel = ttk.Frame(self.root, padding=10)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(self.left_panel, text="Algoritma Parametreleri", font=("Helvetica", 14, "bold")).pack(pady=10)
        ttk.Label(self.left_panel, text="Şehir Sayısı:").pack(anchor=tk.W, pady=2)
        self.spin_cities = ttk.Spinbox(self.left_panel, from_=10, to=100, increment=5)
        self.spin_cities.set(30)
        self.spin_cities.pack(fill=tk.X, pady=5)
        ttk.Label(self.left_panel, text="Popülasyon Boyutu:").pack(anchor=tk.W, pady=2)
        self.spin_pop = ttk.Spinbox(self.left_panel, from_=20, to=500, increment=10)
        self.spin_pop.set(100)
        self.spin_pop.pack(fill=tk.X, pady=5)
        ttk.Label(self.left_panel, text="Mutasyon Oranı:").pack(anchor=tk.W, pady=2)
        self.slider_mut = ttk.Scale(self.left_panel, from_=0.01, to=0.5, value=0.05)
        self.slider_mut.pack(fill=tk.X, pady=5)
        ttk.Label(self.left_panel, text="Elitist Birey Sayısı:").pack(anchor=tk.W, pady=2)
        self.spin_elite = ttk.Spinbox(self.left_panel, from_=0, to=20, increment=1)
        self.spin_elite.set(5)
        self.spin_elite.pack(fill=tk.X, pady=5)
        ttk.Separator(self.left_panel, orient='horizontal').pack(fill=tk.X, pady=15)
        
        self.btn_start = ttk.Button(self.left_panel, text="Start/Continue", command=self.start_simulation)
        self.btn_start.pack(fill=tk.X, pady=5)

        self.btn_pause = ttk.Button(self.left_panel, text="Stop", command=self.pause_simulation)
        self.btn_pause.pack(fill=tk.X, pady=5)

        self.btn_reset = ttk.Button(self.left_panel, text="Restart", command=self.reset_simulation)
        self.btn_reset.pack(fill=tk.X, pady=5)

        ttk.Separator(self.left_panel, orient='horizontal').pack(fill=tk.X, pady=15)
        self.lbl_gen = ttk.Label(self.left_panel, text="Generation: 0", font=("Helvetica", 11))
        self.lbl_gen.pack(anchor=tk.W, pady=2)
        
        self.lbl_dist = ttk.Label(self.left_panel, text="Shortest Path: 0.00", font=("Helvetica", 11, "bold"))
        self.lbl_dist.pack(anchor=tk.W, pady=2)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(9, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_simulation(self):
        self.pause_simulation()

        num_cities = int(self.spin_cities.get())
        pop_size = int(self.spin_pop.get())
        mut_rate = float(self.slider_mut.get())
        elitism = int(self.spin_elite.get())

        self.tsp_prob = TSPProblem(num_cities=num_cities, seed=np.random.randint(1, 10000))
        self.ga = GeneticAlgorithm(
            problem=self.tsp_prob, pop_size=pop_size, mut_rate=mut_rate, elitism_count=elitism,
            selection_func=tournament_selection, crossover_func=order_crossover_ox, mutation_func=swap_mutation
        )
        self.ga.initialize()

        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.set_title("Map")
        coords = self.tsp_prob.city_coords
        self.ax1.scatter(coords[:, 0], coords[:, 1], color='red', s=40, zorder=5)
        self.line, = self.ax1.plot([], [], color='green', linewidth=2, zorder=1)

        self.ax2.set_title("Convergence Curve")
        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Distance")
        self.convergence_line, = self.ax2.plot([], [], color='blue', linewidth=2)
        self.ax2.set_xlim(0, self.max_generations)
        self.ax2.set_ylim(self.ga.best_fitness * 0.3, self.ga.best_fitness * 1.1)
        self.ax2.grid(True)

        self.lbl_gen.config(text="Generation: 0")
        self.lbl_dist.config(text=f"Shortest Path: {self.ga.best_fitness:.2f}")
        
        self.fig.tight_layout()
        self.canvas.draw()

    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.simulation_loop()

    def pause_simulation(self):
        self.is_running = False
        if self.ani_loop:
            self.root.after_cancel(self.ani_loop)
            self.ani_loop = None

    def simulation_loop(self):
        if self.is_running and self.ga.current_iteration < self.max_generations:
            self.ga.step()
            self.ga.current_iteration += 1
            self.ga.history.append(self.ga.best_fitness)

            best_tour = self.ga.best_solution
            ordered_coords = self.tsp_prob.city_coords[best_tour]
            ordered_coords = np.vstack([ordered_coords, ordered_coords]) 
            self.line.set_data(ordered_coords[:, 0], ordered_coords[:, 1])

            self.convergence_line.set_data(range(len(self.ga.history)), self.ga.history)

            self.lbl_gen.config(text=f"Generation: {self.ga.current_iteration}")
            self.lbl_dist.config(text=f"Shortest Path: {self.ga.best_fitness:.2f}")

            self.canvas.draw()
            self.ani_loop = self.root.after(30, self.simulation_loop)
        else:
            self.is_running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPGui(root)
    root.mainloop()
