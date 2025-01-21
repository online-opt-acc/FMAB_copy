"""
functions for tests
"""
import numpy as np
import jax.numpy as jnp


class QuadForm:
    def __init__(self, dim: int):
        self.dim = dim

        self.bias = np.random.randn(dim)
        self.bias1 = np.random.randn()

        mat = np.random.randn(dim, dim)
        self.mat = jnp.dot(mat.T, mat) + np.eye(dim) * 1e-4
    def __call__(self, x):
        assert len(x) == self.dim
        return self.bias1 + (x - self.bias).T @ self.mat @ (x - self.bias)

class LinearFunc:
    def __init__(self, dim: int):
        self.dim = dim

        self.bias = np.random.randn()
        self.weight = np.random.randn(dim)

    def __call__(self, x):
        assert len(x) == self.dim
        return jnp.vdot(self.weight, x) + self.bias

class ModularFunc:
    def __init__(self, dim: int):
        self.dim = dim

        self.bias = np.random.randn() * 2
        self.bias2 = np.abs(np.random.randn()) * 5
        self.weight = np.random.randn(dim)*2

    def __call__(self, x):
        assert len(x) == self.dim
        return jnp.abs(jnp.vdot(self.weight, x) - self.bias) + self.bias2
