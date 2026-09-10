import numpy as np
from abc import ABC, abstractmethod

class CrossOverOperator(ABC):
    def __init__(self,
                 probability,
                 seed: int | None = None):
        super().__init__()

        self.probability = probability
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def crossover(self, X, Y):
        pass

class BinomialCrossOver(CrossOverOperator):
    def __init__(self,
                 probability, seed: int | None = None):
        super().__init__(probability, seed=seed)

    def crossover(self, X, V):
        mask = self.rng.uniform(low=0, high=1, size=(X.shape)) < self.probability
        j_rand = self.rng.integers(low=0, high=X.shape[1], size=X.shape[0])
        mask[np.arange(X.shape[0]), j_rand] = True
        return np.where(mask, V, X)

class SimulatedBinaryCrossover(CrossOverOperator):
    def __init__(self,
                 probability=0.9,
                 eta_c=20,
                 seed: int | None = None):
        super().__init__(probability, seed=seed)
        self.eta_c = eta_c

    def crossover(self, X, Y):
        N, D = X.shape

        cross_mask = self.rng.random(size=(N,1)) <= self.probability

        u = self.rng.random(size=(N,D))
        beta = np.empty((N, D))

        mask_u = u <= 0.5

        beta[mask_u] = (2.0 * u[mask_u]) ** (1.0 / (self.eta_c + 1.0))

        beta[~mask_u] = (1.0 / (2.0 * (1.0 - u[~mask_u]))) ** (1.0 / (self.eta_c + 1.0))
        
        # 4. Cálculo do filho
        # O SBX gera matematicamente 2 filhos. Como precisamos manter o tamanho da matriz N,
        # retornamos o primeiro filho (a variação aleatória de X e Y já garante a mistura).
        offspring = 0.5 * ((1 + beta) * X + (1 - beta) * Y)
        
        # 5. Onde não houve crossover, o pai (X) sobrevive intacto
        return np.where(cross_mask, offspring, X)
