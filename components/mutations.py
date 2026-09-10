import numpy as np
from abc import ABC, abstractmethod

class Mutation(ABC):
    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def mutate(self, X):
        pass

class GaussianMutation(Mutation):
    def __init__(self, sigma, seed: int | None = None):
        super().__init__(seed=seed)
        self.sigma = sigma

    def mutate(self, X):
        noise = self.rng.normal(loc=0, scale=self.sigma, size=X.shape)
        return X + noise

class BitWiseMutation(Mutation):
    def __init__(self, mutation_rate, seed: int | None = None):
        super().__init__(seed=seed)
        self.mutation_rate = mutation_rate

    def mutate(self, X):
        mask = self.rng.random(X.shape) < self.mutation_rate
        return np.bitwise_xor(X, mask).astype(np.int32)

class PolynomialMutation(Mutation):
    def __init__(self, mutation_rate, bounds, eta_m=20, seed: int | None = None):
        super().__init__(seed=seed)
        self.mutation_rate = mutation_rate
        self.bounds = np.array(bounds)
        self.eta_m = eta_m

    def mutate(self, X):
        N, D = X.shape
        
        mut_mask = self.rng.random(size=(N,D)) <= self.mutation_rate
        
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        
        u = self.rng.random(size=(N,D))
        delta_q = np.empty((N, D))

        mask_u = u <= 0.5
        
        delta_q[mask_u] = (2.0 * u[mask_u]) ** (1.0 / (self.eta_m + 1.0)) - 1.0
        delta_q[~mask_u] = 1.0 - (2.0 * (1.0 - u[~mask_u])) ** (1.0 / (self.eta_m + 1.0))

        mutated_X = X + delta_q * (upper - lower)

        return np.where(mut_mask, mutated_X, X)