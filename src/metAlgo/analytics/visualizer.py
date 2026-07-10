import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_curves(history_data):
    plt.figure(figsize=(10, 6))
    
    for entry in history_data:
        name = entry["name"]
        hist = np.array(entry["history"]) 
        if hist.ndim == 0:
            hist = np.full(100, hist) 
            
        plt.plot(hist, label=name)
        
    plt.yscale('log')
    plt.legend()
    plt.show()