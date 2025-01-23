"""
This is a file with online optimizers
"""


from abc import abstractmethod
from typing import List
from collections import defaultdict
from attr import dataclass

from tqdm import tqdm
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


class BanditGradDescent(BaseOnlineOptimizer):
    def __init__(self, vector_function: dict,
                lr_scalers,
                bound_scalers,
                projection_function = lambda x: x
                ):
        """
        params:

            lr_scaler: learning rate scaler D/G
            while running use D_G/sqrt(t)
        """

        super().__init__()

        self.f = vector_function
        self.lr_scalers = lr_scalers
        self.bound_scalers = bound_scalers

        self.projection_function = projection_function

    def run(self, x0: dict, T):
        n_arms = len(self.f)
        losses = defaultdict(list)
        losses_list = []
        pulls = {i: 1 for i in range(n_arms)}
        x_s = x0 #для каждой ручки свои параметры
        x_sums = {i: x.copy() for i, x in x_s.items()}

        values = {i: self.f[i](x_sums[i]) for i in range(n_arms)}
        G = 1

        for t in range(T):
            # select arm
            ucbs = [values[i] - self.bound_scalers[i] * G * (1/pulls[i]**0.5)
                             for i in range(n_arms)]
            arm = np.argmin(ucbs)

            arm_t = pulls[arm]
            f = self.f[arm]
            f_time = pulls[arm]
            x = x_s[arm]

            # получили новую точку
            grad_t = f.grad(x)
            G = max(G, np.linalg.norm(grad_t))
            eta_t = self.lr_scalers[arm]/(f_time**0.5)
            x = x - eta_t * grad_t
            x = self.projection_function(x)
            # обновилли точки
            x_sums[arm] += x
            x_s[arm] = x
            
            pulls[arm] += 1

            x_compute = x_sums[arm]/pulls[arm] # потому что сходимость в среднем
            f_val = f(x_compute)
            loss_t = f_val
            losses[arm].append(loss_t)
            losses_list.append(loss_t)
            values[arm] = f_val
        
        print(G)
        x = {i: x_sums[i]/pulls[i] for i in range(n_arms)}

        return losses, losses_list, x