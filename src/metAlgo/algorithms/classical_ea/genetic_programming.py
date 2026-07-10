import numpy as np
import random
import copy
from metAlgo.framework.base_algorithm import BaseAlgorithm
from metAlgo.framework.base_problem import BaseProblem

class Node:
    def __init__(self, value, children=None, *args, **kwargs):
        self.value = value
        self.children = children if children else []

    def evaluate(self, inputs, functions):
        if callable(self.value):
            args = [child.evaluate(inputs, functions) for child in self.children]
            return self.value(*args)
        elif isinstance(self.value, str) and self.value in inputs:
            return inputs[self.value]
        else:
            return self.value

    def __str__(self):
        if not self.children:
            return str(self.value)
        return f"{self.value.__name__}({', '.join(str(c) for c in self.children)})"

    def all_nodes(self):
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes


def full_method(depth, terminals, functions):
    if depth == 0:
        return Node(random.choice(terminals))
    func = random.choice(functions)
    return Node(func, [full_method(depth - 1, terminals, functions) for _ in range(func.__code__.co_argcount)])



def grow_method(depth, terminals, functions):
    if depth == 0 or (random.random() < 0.5):
        return Node(random.choice(terminals))
    func = random.choice(functions)
    return Node(func, [grow_method(depth - 1, terminals, functions) for _ in range(func.__code__.co_argcount)])


def ramped_half_and_half(max_depth, terminals, functions, population_size):
    population = []
    for depth in range(1, max_depth + 1):
        for _ in range(population_size // (2 * max_depth)):
            population.append(full_method(depth, terminals, functions))
            population.append(grow_method(depth, terminals, functions))
    return population

def subtree_crossover(parent1, parent2):
    p1_copy = copy.deepcopy(parent1)
    p2_copy = copy.deepcopy(parent2)

    node1 = random.choice(p1_copy.all_nodes())
    node2 = random.choice(p2_copy.all_nodes())

    node1.value, node1.children = node2.value, copy.deepcopy(node2.children)
    return p1_copy

def subtree_mutation(tree, terminals, functions, max_depth=3):
    t_copy = copy.deepcopy(tree)
    node = random.choice(t_copy.all_nodes())
    if random.random() < 0.5:
        node.value = random.choice(terminals)
        node.children = []
    else:
        func = random.choice(functions)
        node.value = func
        node.children = [grow_method(max_depth - 1, terminals, functions) for _ in range(func.__code__.co_argcount)]
    return t_copy

class SimpleGP(BaseAlgorithm):
    def __init__(self, problem, pop_size=30, max_depth=3, terminals=None, functions=None, *args, **kwargs):
        super().__init__(problem, name="SimpleGP")  
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.terminals = terminals or ['x0']
        self.functions = functions or ['add', 'sub', 'mul']
        self.history = []

    def initialize(self):
        self.population = [self._random_tree(self.max_depth) for _ in range(self.pop_size)]
        self.fitness = [self.evaluate_tree(tree) for tree in self.population]
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness[0]
    def _random_tree(self, depth):
        if depth == 0 or (depth < self.max_depth and np.random.rand() < 0.3):
         return np.random.choice(self.terminals)
        else:
            func = np.random.choice(self.functions)
            left = self._random_tree(depth - 1)
            right = self._random_tree(depth - 1)
            return [func, left, right]

    def evaluate_tree(self, tree):
        if isinstance(tree, str):
            return 1.0 if tree.isdigit() else 0.0
        
        func, left, right = tree[0], tree[1], tree[2]
        l_val = self.evaluate_tree(left)
        r_val = self.evaluate_tree(right)
        
        if func == 'add': return l_val + r_val
        if func == 'sub': return l_val - r_val
        if func == 'mul': return l_val * r_val
        return 0.0

    def crossover(self, tree1, tree2):
        new_tree = tree1.copy()
        new_tree[1] = tree2[1] 
        return new_tree

    def step(self):
      
        if len(self.population) < 2: return self.best_solution, self.best_fitness 
        
        idx1, idx2 = np.random.choice(len(self.population), 2, replace=False)
        parent1, parent2 = self.population[idx1], self.population[idx2]
        
        offspring = self.crossover(parent1, parent2)
        
        fit_offspring = self.evaluate_tree(offspring)
        if fit_offspring < self.fitness[idx1]:
            self.population[idx1] = offspring
            self.fitness[idx1] = fit_offspring

        self.best_fitness = min(self.fitness)
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.history.append(self.best_fitness)
        return self.best_solution, self.best_fitness
    
    def solve(self, **kwargs):
        if not hasattr(self, 'population') or self.population is None:
            self.initialize()
        return self.best_solution, self.best_fitness

    def run(self):
        self.initialize()
        for _ in range(self.max_iters):
            self.step()
        return self.best_solution, self.best_fitness