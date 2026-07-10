import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, filename="experiment_log.csv", folder="results"):
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filepath = os.path.join(self.folder, filename)
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Algorithm", "Problem", "Iteration", "BestFitness", "MeanFitness"])

    def log(self, algo_name, prob_name, iteration, best_fit, mean_fit):
        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%H:%M:%S"),
                algo_name, 
                prob_name, 
                iteration, 
                f"{best_fit:.6e}", 
                f"{mean_fit:.6e}"
            ])