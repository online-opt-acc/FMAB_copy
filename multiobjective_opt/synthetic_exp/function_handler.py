"""
this is a handler for functions
to wrap them into jax
"""
from typing import Callable, Dict, List

import jax
import numpy as np
from jax import grad, jacobian
from jax import numpy as jnp


class DefalutRandomizer:
    def __init__(self, dim=0, **kwargs):
        self.dim = dim

    def get_sigma(self):
        return 0

    def __call__(self, *args, **kwds):
        if self.dim == 0:
            return 0.0
        return np.zeros(self.dim, dtype=float)


class NormalRandomizer(DefalutRandomizer):
    def __init__(self, dim, sigma, seed=None):
        """
        returns noise to be scaled as
        E|| G - g ||^2 <= sigma^2
        """
        if isinstance(dim, int):
            if dim == 0:
                dim = 1
            dim = (dim,)

        self.dim = dim
        self.sigma = sigma
        self.rand = np.random.default_rng(seed)

    def get_sigma(self):
        return self.sigma

    def __call__(self):
        return self.rand.normal(0, self.sigma / (np.prod(self.dim) ** 0.5), self.dim)


class JaxFunc:
    """
    scalar function wrapper
    """

    def __init__(
        self,
        func: Callable,
        input_dim,
        output_dim=1,
        *,
        grad_kwargs: Dict = None,
        device: str = "cpu",
        value_randomizer=None,
        grad_randomizer=None,
    ) -> None:
        """
        grad_kwargs: {argnums}
        """
        self.func = func

        if grad_kwargs is None:
            grad_kwargs = {}

        assert output_dim >= 1
        if output_dim > 1:
            self._grad = jacobian(func, **grad_kwargs)
        else:
            self._grad = grad(func, **grad_kwargs)

        if value_randomizer is None:
            value_randomizer = DefalutRandomizer(output_dim)
        if grad_randomizer is None:
            grad_randomizer = DefalutRandomizer((input_dim, output_dim))

        self.value_randomizer = value_randomizer
        self.grad_randomizer = grad_randomizer

        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        jax.config.update("jax_platform_name", device)

    def __call__(self, *x):
        res = self.func(*x) + self.value_randomizer()
        return res.squeeze()

    def grad(self, *x):
        try:
            # res = self._grad(*x) + self.grad_randomizer()
            # print(self._grad(*x).shape, self.grad_randomizer().shape)
            return self._grad(*x) + self.grad_randomizer().squeeze()
        except TypeError as e:
            # TODO: ловить ошибку если функция векторная и переделать ее градиент
            raise e


# def stack_functions(
#     functions: List[JaxFunc], grad_kwargs: Dict = None, device="cpu"
# ) -> JaxFunc:
#     if grad_kwargs is None:
#         grad_kwargs = {}
#     assert device == "cpu" or device.startswith("cuda")

#     functions = [(f.func if isinstance(f, JaxFunc) else f) for f in functions]

#     def function(x):
#         rez = jnp.array([f(x) for f in functions])
#         return rez

#     vec_func = JaxFunc(
#         function, vector_valued=True, device=device, grad_kwargs=grad_kwargs
#     )
#     return vec_func
