"""
this is a handler for functions 
to wrap them into jax
"""
from shutil import ExecError
from typing import Callable, List, Dict

import jax
from jax import  numpy as jnp
from jax import grad, jacobian, hessian
from numpy import isin


class JaxFunc:
    """
    scalar function wrapper
    """
    def __init__(self, func: Callable,
                    vector_valued = False,*,
                    grad_kwargs: Dict = None,
                    device: str = "cpu") -> None:
        """
        grad_kwargs: {argnums}
        """
        self.func = func
        
        if grad_kwargs is None:
            grad_kwargs = {}

        if vector_valued:
            self._grad = jacobian(func, **grad_kwargs)
        else:
            self._grad = grad(func, **grad_kwargs)


        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        jax.config.update('jax_platform_name', device)

    def __call__(self, *x):
        return self.func(*x)

    def grad(self, *x):
        try:
            return self._grad(*x)
        except TypeError as e:
            # TODO: ловить ошибку если функция векторная и переделать ее градиент
            raise e

def stack_functions(functions: List[JaxFunc], grad_kwargs: Dict = {}, device = "cpu") -> JaxFunc:
    assert device == "cpu" or device.startswith("cuda")

    functions = [(f.func if isinstance(f, JaxFunc) else f) for f in functions]

    def function(x):
        rez = jnp.array([f(x) for f in functions])
        return rez

    vec_func = JaxFunc(function, vector_valued=True, device=device,
                grad_kwargs=grad_kwargs)
    return vec_func