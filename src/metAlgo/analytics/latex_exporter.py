from typing import List, Dict
import numpy as np

def generate_ieee_latex_table(results_list: List[Dict[str, any]], problem_name: str) -> str:
    latex_code = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Statistical Results on {problem_name} Benchmark (30 Independent Runs)}}",
        "\\label{tab:optimization_results}",
        "\\begin{tabular}{l c c c c}",
        "\\hline\\hline",
        "\\textbf{Algorithm} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{Best} & \\textbf{Worst} \\\\",
        "\\hline"
    ]
    
    for res in results_list:
        algo_name = res["Algorithm"]
        mean_val = f"{res['Mean']:.4e}"
        std_val = f"{res['Std']:.4e}"
        best_val = f"{res['Best']:.4e}"
        worst_val = f"{res['Worst']:.4e}"
        
        row = f"{algo_name} & {mean_val} & {std_val} & {best_val} & {worst_val} \\\\"
        latex_code.append(row)
        
    latex_code.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

def format_value(value):
    if isinstance(value, np.ndarray):
        value = float(value.min()) 
    return "{:.4e}".format(value)