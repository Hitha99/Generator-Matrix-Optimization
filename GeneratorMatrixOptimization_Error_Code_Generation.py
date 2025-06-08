# -*- coding: utf-8 -*-
"""
This program searches for generator matrices with minimal m-height using differential evolution.

The procedure begins with a generator matrix that results from the union of an identity matrix and submatrix P. CodeHeightCalculator class validates the matrix to ensure that it contains an identity submatrix, all integer elements, values within range, and no zero columns.

The code computes the maximal m-height for all matrices by iterating over row and column pairs, selecting index subsets, generating sign vectors, and solving a linear programming problem for each case. The worst-case result of the combinations is the m-height of the matrix.

The optimization is via the differential evolution algorithm, evolving random candidate matrices over generations. Mutation, crossover, and selection provide new solutions. The fitness is the m-height for every candidate, and the optimizer chooses matrices of lesser value.

The CodeResultsManager caches and recalls computed matrices and their m-heights to prevent recomputation and enhance efficiency.

The principal function establishes parameter values for n, k, and m. It checks for existing results, runs the optimizer if necessary, and caches optimal matrices and their m-heights, completing the search without duplicates.
"""

import numpy as np
import itertools
from scipy.optimize import linprog
from datetime import datetime
from tqdm import tqdm
import pickle
import os
import logging
import json


class CodeHeightCalculator:
    def __init__(self, G: np.ndarray, m: int, verbose: bool = False):
        self.G = np.array(G, dtype=np.float64)
        self.k, self.n = self.G.shape
        self.m = m
        self.verbose = verbose
        if verbose:
            self._check_matrix_structure()

    #This function checks the proper form of the generator matrix.
    #It verifies whether the first k columns constitute an identity matrix, integers in the remaining portion
    #of the entries within the specified limit, and the absence of zero columns.
    #It also verifies the condition that m should not exceed n minus k.
    def _check_matrix_structure(self):
        check_identity_matrix = np.allclose(self.G[:, :self.k], np.eye(self.k))
        print(f"Identity submatrix check (first k columns form Iₖ): {'Passed' if check_identity_matrix else 'Failed'}")

        P = self.G[:, self.k:]
        print(f"Integer values check (all entries in P are integers): {'Passed' if np.all(P == P.astype(int)) else 'Failed'}")
        print(f"Check the range (entries of P lie within [-100, 100]): {'Passed' if np.all((P >= -100) & (P <= 100)) else 'Failed'}")
        print(f"Zero column check (no column in P is entirely zero): {'Passed' if not np.any(np.all(P == 0, axis=0)) else 'Failed'}")

        print(f"\nMatrix parameters: n = {self.n}, k = {self.k}, m = {self.m}")
        print(f"Constraint verification (m ≤ n - k): {'Passed' if self.m <= self.n - self.k else 'Failed'}")

    #This function is specifically aimed at computing the maximum m-height for a given matrix.
    #It systematically goes through all possible pairs of row and column indices,
    #creating different sets of alternate indices and sign patterns for each.
    #For every such distinct setup, it goes on to solve linear programming problems.
    #Finally, the function returns the maximum value found over all the various configurations as the final m-height result.
    def compute_mheight(self):
        max_height = 0
        for p in range(self.n):
            for q in range(self.n):
                if p == q:
                    continue
                rem = [j for j in range(self.n) if j not in {p, q}]
                for X in itertools.combinations(rem, self.m - 1):
                    for psi in itertools.product([-1, 1], repeat=self.m):
                        tau_inv = self._create_inverse(p, q, X)
                        height = self._solve_lp(p, q, X, psi, tau_inv)
                        if height == float('inf'):
                            return float('inf')
                        max_height = max(max_height, height)
        return max_height

    #This particular helper function is tasked with developing an overall mapping that connects the
    #new positions of the reordered selected indices to their corresponding original positions.
    #This mapping is essential and is utilized in order to arrange the variables in the correct
    #way as part of the process of generating the linear programming constraints.
    def _create_inverse(self, p, q, X):
        tau = [p] + sorted(X) + [q] + sorted(set(range(self.n)) - {p, q} - set(X))
        return {j: i for i, j in enumerate(tau)}

    #This function is specifically used to formulate and solve the linear program for a given arrangement of rows,
    #columns, subset indices, and sign patterns. In doing this, it carefully constructs the cost vector and inequality
    #and equality constraints, then solves the linear program to determine the contribution to the m-height.
    #If it so occurs that the linear program is infeasible or unbounded in its outcome, the function will return infinity.
    def _solve_lp(self, row_idx, col_idx, X, psi, tau_inv):
        k = self.k
        c = [-psi[0] * self.G[i, row_idx] for i in range(k)]
        A = []
        b = []
        for j in X:
            s_j = psi[tau_inv[j]]
            A.append([s_j * self.G[i, j] - psi[0] * self.G[i, row_idx] for i in range(k)])
            A.append([-s_j * self.G[i, j] for i in range(k)])
            b.extend([0, 0])

        Y = set(range(self.n)) - {row_idx, col_idx} - set(X)
        for j in Y:
            A.append([self.G[i, j] for i in range(k)])
            A.append([-self.G[i, j] for i in range(k)])
            b.extend([1, 1])

        A_eq = [[self.G[i, col_idx] for i in range(k)]]
        b_eq = [1]
        res = linprog(c=c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method='highs')
        if res.success:
            return -res.fun
        return float('inf') if res.status in [3, 4] else 0


class CodeResultsManager:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.matrix_file = f"generatorMatrix-{identifier}.pkl"
        self.height_file = f"mHeight-{identifier}.pkl"

    #This function loads previously computed generator matrices and
    #their corresponding m-heights from pickle files. If the files are not found, it initializes empty dictionaries.
    def load(self):
        if os.path.exists(self.matrix_file):
            with open(self.matrix_file, 'rb') as f:
                matrix_data = pickle.load(f)
        else:
            matrix_data = {}

        if os.path.exists(self.height_file):
            with open(self.height_file, 'rb') as f:
                height_data = pickle.load(f)
        else:
            height_data = {}

        return matrix_data, height_data

    def save(self, matrices, heights):
        with open(self.matrix_file, 'wb') as f:
            pickle.dump(matrices, f)
        with open(self.height_file, 'wb') as f:
            pickle.dump(heights, f)

#This helper function creates the entire generator matrix by horizontally
#concatenating the identity matrix and submatrix P. The resulting systematic matrix is used in m-height calculations.
def create_systematic_matrix(k, P):
    return np.hstack((np.eye(k), P))

#This particular function is designed to scan the matrix P
#to determine whether it has any column that contains all its values as zero.
def check_zero_column(P):
    return np.any(np.all(P == 0, axis=0))

#This function calculates the m-height of a matrix and also deals with errors.
#If the matrix is invalid, for example, having zero columns, it returns a penalty value of 1e10 instead of calculating.
def safe_calculate_mheight(P, k, n, m):
    if check_zero_column(P):
        return 1e10
    try:
        G = create_systematic_matrix(k, P)
        return CodeHeightCalculator(G, m).compute_mheight()
    except Exception:
        return 1e10

#This method sets up logging for the optimizer, creating a logger that displays progress
#and output on the console and a log file to monitor the process.
def setup_logger(n, k, m):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"differential_evolution_log_n{n}_k{k}_m{m}_{timestamp}.log"
    logger = logging.getLogger(f"DE_Logger_n{n}_k{k}_m{m}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# This function uses the Differential Evolution algorithm to find the minimum m-height generator matrix
#by evolving candidate matrices through mutation, crossover, and selection, keeping track of the best solution over generations.
def differential_evolution_optimizer(k, n, m, pop_size=50, generations=100, F=0.8, CR=0.9, value_range=20):
    if m > n - k:
        raise ValueError(f"Infeasible configuration: m ({m}) > n - k ({n - k})")
    logger = setup_logger(n, k, m)
    logger.info(f"Starting Differential Evolution for n={n}, k={k}, m={m}")

    dim = k * (n - k)
    bounds = (-value_range, value_range)
    population = [np.random.randint(bounds[0], bounds[1] + 1, size=dim) for _ in range(pop_size)]
    fitness = [safe_calculate_mheight(ind.reshape(k, n - k), k, n, m) for ind in population]

    best_idx = np.argmin(fitness)
    best_vector = population[best_idx]
    best_fitness = fitness[best_idx]

    for gen in range(generations):
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = np.clip(np.round(mutant), bounds[0], bounds[1])
            trial = np.where((np.random.rand(dim) < CR) | (np.arange(dim) == np.random.randint(0, dim)), mutant, population[i])
            P_trial = trial.reshape(k, n - k)
            trial_fitness = safe_calculate_mheight(P_trial, k, n, m)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_vector = trial
        logger.info(f"Generation {gen + 1}: Best m-height = {best_fitness}")
        logger.info(f"Best matrix P so far:\n{best_vector.reshape(k, n - k)}")

    return best_vector.reshape(k, n - k), best_fitness

#he main function enumerates parameter combinations to investigate,
#loading prior results to streamline missing configurations.
#It executes the optimizer and persists results following every computation.
def main():
    identifier = "File"
    parameter_list = [
        (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
        (9, 5, 2), (9, 5, 3), (9, 5, 4),
        (9, 6, 2), (9, 6, 3),
        (10, 4, 2), (10, 4, 3), (10, 4, 4), (10, 4, 5), (10, 4, 6),
        (10, 5, 2), (10, 5, 3), (10, 5, 4), (10, 5, 5),
        (10, 6, 2), (10, 6, 3), (10, 6, 4)
    ]
    results_mgr = CodeResultsManager(identifier)
    matrices, heights = results_mgr.load()
    for n, k, m in parameter_list:
        key = json.dumps([n, k, m])
        if key in matrices:
            continue
        best_P, best_h = differential_evolution_optimizer(k, n, m, pop_size=30, generations=1000)
        if best_h < 1e10:
            matrices[key] = best_P
            heights[key] = best_h
            results_mgr.save(matrices, heights)

if __name__ == '__main__':
    main()