"""
functions for tests
"""
import jax.numpy as jnp
import numpy as np


class BaseFunc:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, *args, **kwds):
        raise NotImplementedError


class QuadForm(BaseFunc):
    def __init__(self, dim: int):
        super(QuadForm, self).__init__(dim)
        self.dim = dim

        self.bias = np.random.randn(dim)
        self.bias1 = np.random.randn()

        mat = np.random.randn(dim, dim)
        self.mat = jnp.dot(mat.T, mat) + np.eye(dim) * 1e-4

    def __call__(self, x):
        assert len(x) == self.dim
        return self.bias1 + jnp.sqrt((x - self.bias).T @ self.mat @ (x - self.bias))


class QuadFormNotStrongly(BaseFunc):
    def __init__(self, dim: int, min_val=None):
        super(QuadFormNotStrongly, self).__init__(dim)
        self.dim = dim

        self.bias = np.zeros(dim)  # np.random.randn(dim)

        if min_val is None:
            min_val = np.random.randn()

        self.bias1 = min_val

        self.weights = np.random.randn(dim)
        self.eigenvalues = np.exp(np.random.rand(dim))
        self.eigenvalues[1:] *= np.random.rand(dim - 1) > 0.5
        self.mat = np.diag(self.eigenvalues)

    def get_params(self, x):
        L = max(self.eigenvalues)
        R = np.linalg.norm(x - self.bias)
        return L, R

    def __call__(self, x):
        assert len(x) == self.dim
        return (
            self.bias1
            + 0.5 * (x - self.bias).T @ self.mat @ (x - self.bias)
            + jnp.sqrt(1 + jnp.vdot(self.weights, (x - self.bias)) ** 2)
            - 1
        )


class LinearFunc:
    def __init__(self, dim: int):
        self.dim = dim

        self.bias = np.random.randn()
        self.weight = np.random.randn(dim)

    def __call__(self, x):
        assert len(x) == self.dim
        return jnp.vdot(self.weight, x) + self.bias


class ModularFunc(BaseFunc):
    def __init__(self, dim: int, min_val=None, num_planes=None):
        super(ModularFunc, self).__init__(dim)
        self.dim = dim

        self.bias = np.ones(dim)  # np.random.rand(dim) * 2 - 1

        if num_planes is None:
            num_planes = dim

        if min_val is None:
            min_val = np.abs(np.random.randn())

        self.bias2 = min_val

        self.weight = np.random.randn(num_planes, dim)

    def get_params(self, *args):
        L = np.max(np.abs(self.weight))
        # R = 2 * np.linalg.norm(x)
        return L, None

    def __call__(self, x):
        assert len(x) == self.dim
        return jnp.abs(jnp.max(jnp.dot(self.weight, x - self.bias))) + self.bias2


class RosenbrockSkokov:
    def __init__(self, dim: int, min_val=None):
        if min_val is None:
            min_val = np.random.randn()
        self.min_val = min_val
        self.dim = dim
        self.min_point = np.ones(self.dim, dtype=float)

    def get_params(self, x):
        L = max(self.eigenvalues)
        R = np.linalg.norm(x - self.min_point)
        return L, R

    def __call__(self, x):
        res = 1 / 4 * (x[0] - 1) ** 2
        for i in range(len(x) - 1):
            res += (x[i + 1] - 2 * x[i] ** 2 + 1) ** 2
        return res


class RosenbrockNonsmooth:
    def __init__(self, dim: int, min_val=None):
        if min_val is None:
            min_val = np.random.randn()
        self.min_val = min_val
        self.dim = dim

    def get_params(self, x):
        L = 4
        # R = np.linalg.norm(x - self.min_point)
        return L, None

    def __call__(self, x):
        res = 1 / 4 * jnp.abs(x[0] - 1)
        for i in range(len(x) - 1):
            res += jnp.abs(x[i + 1] - 2 * x[i] ** 2 + 1)
        return res
