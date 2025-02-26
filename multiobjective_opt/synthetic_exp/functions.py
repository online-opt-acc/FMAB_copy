"""
functions for tests
"""
import jax.numpy as jnp
import numpy as np
from jax import grad
from scipy.optimize import minimize, linprog
from tqdm import TMonitor
# from torch import T


class BaseFunc:
    def __init__(self, dim, seed = 0):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, *args, **kwds):
        raise NotImplementedError


class QuadForm(BaseFunc):
    def __init__(self, dim: int, seed = 0):
        super(QuadForm, self).__init__(dim, seed)
        self.dim = dim
        
        self.bias = self.rng.normal(size = (dim,))
        self.bias1 = self.rng.normal()

        mat = self.rng.normal(size=(dim, dim))
        self.mat = jnp.dot(mat.T, mat) + np.eye(dim) * 1e-4

    def __call__(self, x):
        assert len(x) == self.dim
        return self.bias1 + jnp.sqrt((x - self.bias).T @ self.mat @ (x - self.bias))


class QuadFormSQRT(BaseFunc):
    def __init__(self, dim: int, min_val=None, seed = 0, eigenvalues = None, eigval_pow = 5):
        super(QuadFormSQRT, self).__init__(dim, seed)
        self.dim = dim

        self.bias =  self.rng.normal(size = (dim, ))

        if min_val is None:
            min_val = self.rng.normal()

        self.bias1 = min_val

        if eigenvalues is None:
            eigenvalues = np.exp(- eigval_pow * self.rng.random(dim))
            eigenvalues[0] = 1
        self.eigenvalues = eigenvalues

        self.mat = np.diag(self.eigenvalues)

    def get_params(self, x):
        L = max(self.eigenvalues)
        R = np.linalg.norm(x - self.bias)
        return L, R

    def __call__(self, x):
        assert len(x) == self.dim
        return (
            self.bias1 - 1 + jnp.sqrt(1 + (x - self.bias).T @ self.mat @ (x - self.bias))
        )



class ModularFunc(BaseFunc):
    def __init__(self, dim: int, optimization_set, min_val=None, num_planes=None, seed = 0):
        """
        since function is not necessarily bounded from below, there optimization_set should be provided
        """
        super(ModularFunc, self).__init__(dim, seed=seed)

        self.dim = dim

        if num_planes is None:
            num_planes = dim
        self.num_planes = num_planes
        self.bias = self.rng.normal(size = (num_planes,)) * 2  # np.random.rand(dim) * 2 - 1
        self.weight = self.rng.normal(size=(num_planes, dim)) * 2

        if min_val is None:
            min_val = np.abs(self.rng.normal())

        x0 = np.zeros(dim, float)
        real_min = self._get_real_min(x0, optimization_set)
        print(real_min)
        self.bias2 = min_val - real_min

    def _get_real_min(self, x0, bounds):
        t_vector = -np.ones(shape=(self.num_planes, 1))
        a_tmp = np.concat([self.weight, t_vector ], 1)
        b_tmp = -self.bias
        c_tmp = np.zeros((self.dim + 1,), dtype=float)
        c_tmp[-1] = 1.
        x0 = np.concat([x0, [1]])
        bounds = [(l, u) for l, u in zip(bounds.lb, bounds.ub)] + [(None, None)]
        val = linprog(c_tmp, A_ub= a_tmp, b_ub=b_tmp, x0=x0, bounds=bounds)
        self.x0 = val.x[:-1]
        res_val = val.x[-1]
        return res_val

    def get_params(self, x):
        L = np.max(np.abs(self.weight))
        R = np.linalg.norm(x - self.x0)
        return L, R

    def __call__(self, x):
        assert len(x) == self.dim
        return jnp.max(jnp.dot(self.weight, x) + self.bias) + self.bias2

