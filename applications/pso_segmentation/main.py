
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../../src")
sys.path.insert(0, src_path)
from metAlgo.framework.base_problem import BaseProblem
from metAlgo.algorithms.swarm_and_physics.particle_swarm import ParticleSwarmOptimization

class PSOSegmentationProblem(BaseProblem):
    def __init__(self, image_path: str, num_thresholds: int = 3):
       
        bounds = [(0.0, 255.0) for _ in range(num_thresholds)]
        super().__init__(dim=num_thresholds, bounds=bounds, name="OtsuSegmentation")
        
        self.original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        hist = cv2.calcHist([self.original_img], [0], None, [256], [0, 256]).ravel()
        self.hist = hist / hist.sum()
        self.intensities = np.arange(256)
        self.global_mean = np.sum(self.intensities * self.hist)

    def evaluate(self, x: np.ndarray) -> float:
       
        th = np.sort(np.round(x).astype(int))
        th = np.clip(th, 0, 255)
        
        boundaries = [0] + list(th) + [256]
        variance = 0.0
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if start >= end: continue
            
            class_prob = self.hist[start:end]
            w = np.sum(class_prob)
            
            if w > 0:
                class_intensities = self.intensities[start:end]
                mu = np.sum(class_intensities * class_prob) / w
                variance += w * ((mu - self.global_mean) ** 2)
                
        return -float(variance)

    def apply_segmentation(self, thresholds: np.ndarray) -> np.ndarray:

        th = np.sort(np.round(thresholds).astype(int))
        th = np.clip(th, 0, 255)
    
        segmented_classes = np.digitize(self.original_img, th)
        num_classes = len(th) + 1
        step = 255 // (num_classes - 1) if num_classes > 1 else 255
        return (segmented_classes * step).astype(np.uint8)

def run_app():
    
    img_path = "test_image.png"
    if not os.path.exists(img_path):
        x = np.linspace(0, 255, 300, dtype=np.uint8)
        img = np.tile(x, (300, 1))
        noise = np.random.normal(0, 20, img.shape).astype(np.int16)
        cv2.imwrite(img_path, np.clip(img + noise, 0, 255).astype(np.uint8))
    
    problem = PSOSegmentationProblem(img_path, num_thresholds=3)
    
    pso = ParticleSwarmOptimization(problem, pop_size=30)
    best_sol, best_fit = pso.solve(max_iterations=50)
    
    print(f"[SUCCESS] Optimal Thresholds Found: {np.sort(np.round(best_sol).astype(int))}")
    print(f"Max Variance: {-best_fit:.2f}")

    seg_img = problem.apply_segmentation(best_sol)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(problem.original_img, cmap='gray')
    ax1.set_title("Orijinal Resim")
    ax1.axis('off')
    
    ax2.imshow(seg_img, cmap='jet')
    ax2.set_title("PSO Segmentasyonu")
    ax2.axis('off')
    
    ax3.plot([-h for h in pso.history], color='red', linewidth=2)
    ax3.set_title("Sürü Yakınsama Grafiği (Varyans)")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_app()