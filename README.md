# metAlgo

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Continuous Integration](https://github.com/archiristo/metAlgo/actions/workflows/ci.yml/badge.svg)

## About The Project

`metAlgo` (Meta-Heuristic Optimization Framework) is an extensible optimization library containing over 25 popular meta-heuristic algorithms and providing academic-standard statistical analyses.

## 🚀 Key Features
- **Modular Design:** Adding new algorithms is easy thanks to the `BaseAlgorithm` and `BaseProblem` classes.
- **Academic Reporting:** Automated experimental results with IEEE format LaTeX tables and Friedman statistical tests.
- **Powerful Arena:** An experimental arena that automatically runs multiple problem and algorithm combinations.
- **Tested:** 100% coverage with 148+ unit tests.

## 📦 Installation

Create a local copy of the project:
```bash
git clone [https://github.com/archiristo/metAlgo.git](https://github.com/archiristo/metAlgo.git)
cd metAlgo
```
Upload the dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Usage
To run the experiment arena and test all algorithms on benchmark problems:

```bash
python run_experiment.py
```

## Adding a New Algorithm
Create a new file in the src/metAlgo/algorithms/ directory and inherit from the BaseAlgorithm class:

```python
from metAlgo.framework.base_algorithm import BaseAlgorithm

class MyNewAlgo(BaseAlgorithm):
    def initialize(self):
        # Initial Logic
        pass

    def step(self):
        # A logic of one iteration
        pass
```

## 🛠 Tests
Use pytest to test the stability of the framework:

```bash
pytest tests/
```

## 🤝 Contributing
Your contributions are valuable to us! 
Please:
-Submit your suggestions by opening an issue.
-Develop new features in your own feature/branch.
-Make sure you pass all tests with pytest before making changes.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
archiristo | 2026