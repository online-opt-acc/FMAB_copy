"""
here is a generators for w
"""

from abc import abstractmethod

import numpy as np

class WGenerator:
    def __init__(self, dim: int, T: int):
        self.dim = dim
        self.T = T
    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError()

class StochasticWGenerator(WGenerator):
    def __init__(self, dim, T, seed = 0):
        super().__init__(dim, T)
        self.rand_generator = np.random.default_rng(seed)

    def __call__(self, *args, **kwds):
        for _ in range(self.T):
            rand_vec = np.abs(self.rand_generator.normal(size=self.dim))
            rand_vec = rand_vec/np.sum(rand_vec)
            yield rand_vec