from abc import ABC, abstractmethod
from typing import Literal
from warnings import warn

import numpy as np
from scipy.stats import qmc


class PopulationOptimizer(ABC):

    def __init__(self,
                 problem,
                 direction,
                 *,
                 population_size,
                 dimensions,
                 bounds,
                 sampling: Literal["uniform", "lhs", "choice"] = "uniform",
                 seed: int | None = None,
                 **kwargs):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.problem = problem
        self.direction = direction.lower()
        if self.direction not in ['max', 'min']:
            warn("Direction not specified. Must be either 'min' or 'max'. Defaulting to 'min'")
            self.direction = 'min'
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = np.array(bounds)
        if sampling.lower() == "uniform":
            self.X = self.rng.uniform(low=self.bounds[:,0], high=self.bounds[:,1], size=(population_size, dimensions))
        elif sampling.lower() == "lhs":
            lhs = qmc.LatinHypercube(d=dimensions, seed=42)
            self.X = (
                lhs.random(n=population_size) * (self.bounds[:,1] - self.bounds[:,0])
                + self.bounds[:,0]
            )
        elif sampling.lower() == "choice":
            # Assuming the bounds are a 2D array [[lower_bound, upper_bound]]
            xl, xu = self.bounds[0][0], self.bounds[0][1]
            self.X = self.rng.choice(
                np.arange(xl, xu),
                size=(population_size, dimensions),
                replace=True,
            )
        else:
            raise ValueError("Invalid initialization method. Must be either 'uniform' or 'lhs'.")
        self.fitness = self.problem.evaluate(self.X)

    @abstractmethod
    def evolve(self):
        pass

    def get_population(self):
        return self.X, self.fitness

    def get_best(self):
        if self.direction == 'max':
            idx = np.argmax(self.fitness)
        else:
            idx = np.argmin(self.fitness)
        return self.X[idx], self.fitness[idx]

class MultiObjectiveOptimizer(PopulationOptimizer):
    def __init__(self,
                 problem,
                 *,
                 population_size,
                 dimensions,
                 bounds,
                 direction: str = 'min',
                 **kwargs):
        super().__init__(problem,
                         direction=direction,
                         population_size=population_size,
                         dimensions=dimensions,
                         bounds=bounds,
                         **kwargs)

    def fast_non_dominated_sort(self, fitness):
        """
        Retorna uma lista de listas contendo os índices dos indivíduos em cada frente.
        Ex: [[idx_frente_1], [idx_frente_2], ...]
        """
        N = fitness.shape[0]
        
        # Using broadcasting to check for dominance
        # diff[i, j, m] = fitness[i, m] - fitness[j, m]
        # 'diff' has shape (N, N, M), where m corresponds to the number of objectives
        diff = fitness[:, np.newaxis, :] - fitness[np.newaxis, :, :]
        
        # 'dominates' is a (N, N) matrix. Index (i,j) = True means that
        # solution 'i' dominates solution 'j'
        dominates = np.logical_and(np.all(diff <= 0, axis=-1), np.any(diff < 0, axis=-1))
        
        # The index 'j' of n_p represents how many other solutions dominate solution 'j'
        n_p = np.sum(dominates, axis=0)
        
        # Building dominated sets. Index 'i' of S_p contains the index of what other
        # solutions are dominated by solution 'i'.
        S_p = [np.where(dominates[i])[0].tolist() for i in range(N)]
        
        # Building the fronts
        fronts = []
        current_front = np.where(n_p == 0)[0].tolist()
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            
            for p in current_front:
                for q in S_p[p]:
                    n_p[q] -= 1 
                    if n_p[q] == 0:
                        next_front.append(q)
                        
            current_front = next_front
            
        return fronts

    def get_best(self):
        fronts = self.fast_non_dominated_sort(self.fitness)
        if not fronts:
            return np.array([]), np.array([])
        pareto_front_idx = fronts[0]
        return self.X[pareto_front_idx], self.fitness[pareto_front_idx]
