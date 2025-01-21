"""
This is a file with online optimizers
"""


from abc import abstractmethod
from typing import List
from tqdm import tqdm

from attr import dataclass
import numpy as np
import jax.numpy as jnp
from scipy.optimize import Bounds, minimize

from .w_generators import WGenerator
from .function_handler import JaxFunc

@dataclass
class OptmizerReturn:
    x: np.array
    losses: List[float]
    w_mean: np.array


class BaseOnlineOptimizer:
    @abstractmethod
    def run(self, x0,):
        ...


class OnlineGradientDescent(BaseOnlineOptimizer):
    def __init__(self, function,
                w_generator: WGenerator,
                lr_scaler = 0.1,
                projection_function = lambda x: x
                ):
        """
        params:

            lr_scaler: learning rate scaler D/G
            while running use D_G/sqrt(t)
        """

        super().__init__()

        self.f = function
        self.lr_scaler = lr_scaler

        self.T = w_generator.T
        self.w_dim = w_generator.dim
        self.w_generator = w_generator
        self.projection_function = projection_function

    def run(self, x0):
        losses = []
        w_sum = np.zeros(self.w_dim)
        x = x0

        for t, w_t in enumerate(self.w_generator(), start = 1):

            f_val = self.f(w_t, x)
            loss_t = f_val
            grad_t = self.f.grad(w_t, x)

            eta_t = self.lr_scaler/(t**0.5)
            x = x - eta_t * grad_t
            x = self.projection_function(x)

            losses.append(loss_t)
            w_sum += w_t

        losses = np.cumsum(losses)
        w_sum /= self.T
        return OptmizerReturn(x, losses, w_sum)
    

class RFTL(BaseOnlineOptimizer):
    def __init__(self, function,
                w_generator: WGenerator,
                eta = 0.1,
                projection_function = lambda x: x
                ):
        """
        params:

            eta: learning rate scaler D/G
            while running use D_G/sqrt(t)
        """

        super().__init__()

        self.f = function
        self.eta = eta

        self.T = w_generator.T
        self.w_dim = w_generator.dim
        self.w_generator = w_generator
        self.projection_function = projection_function

    def minimize(self, x , f, **kwargs):
        val = minimize(f,
                    x0=x,
                    **kwargs
                    )
        return val.x
    
    def run(self, x0, bounds = None):
        
        losses = []
        w_sum = np.zeros(self.w_dim)
        
        grad_accumulator = np.zeros(x0.shape)

        reg_function = JaxFunc(lambda x: 0.5 * jnp.vdot(x, x),)

        x = self.minimize(x0, reg_function, jac = reg_function.grad, bounds = bounds)

        for t, w_t in tqdm(enumerate(self.w_generator(), start = 1)):

            f_val = self.f(w_t, x)
            loss_t = f_val
            grad_t = self.f.grad(w_t, x)

            grad_accumulator += grad_t

            func_t = JaxFunc(lambda x: self.eta * jnp.vdot(grad_accumulator, x) + reg_function(x))
            grad_f_t = func_t.grad

            x = self.minimize(x, func_t, jac = grad_f_t, bounds=bounds)

            losses.append(loss_t)
            w_sum += w_t

        losses = np.cumsum(losses)
        w_sum /= self.T
        return OptmizerReturn(x, losses, w_sum)


class RFTL_structured(RFTL):    
    def run(self, x0, vector_func, bounds = None):
        
        losses = []
        w_sum = np.zeros(self.w_dim)
        
        grad_accumulator = np.zeros(x0.shape)
        reg_function = JaxFunc(lambda x: 0.5 * jnp.vdot(x, x),)

        x = self.minimize(x0, reg_function, jac = reg_function.grad, bounds = bounds)

        for t, w_t in tqdm(enumerate(self.w_generator(), start = 1)):

            f_val = self.f(w_t, x)
            loss_t = f_val
            grad_t = self.f.grad(w_t, x)

            grad_accumulator += grad_t
            # print( vector_func.grad(x).shape)
            func_t = JaxFunc(lambda x:
                            self.eta * jnp.vdot(w_sum, vector_func(x)) + reg_function(x))
            grad_f_t = func_t.grad

            x = self.minimize(x, func_t, jac = grad_f_t, bounds=bounds)

            losses.append(loss_t)
            w_sum += w_t

        losses = np.cumsum(losses)
        w_sum /= self.T
        return OptmizerReturn(x, losses, w_sum)

