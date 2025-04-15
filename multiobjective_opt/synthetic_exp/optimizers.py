from abc import abstractmethod
import copy

import jax.numpy as jnp
import numpy as np
from jax import grad
from multiobjective_opt.synthetic_exp.function_handler import JaxFunc
from scipy.optimize import Bounds, minimize

class Constraints:
    def __init__(self, bounds: Bounds):
        self.bounds = bounds

    def __call__(self, s, x0=None):
        s = np.array(s)
        d = len(s)
        bounds = self.bounds
        if x0 is None:
            x0 = np.zeros(d)

        def f(x):
            return jnp.vdot(s, x) + 1 / 2 * jnp.vdot(x - x0, x - x0)

        val = minimize(f, x0=x0, bounds=bounds, jac=grad(f))

        minimizer = val.x
        return minimizer


class BaseOptimizer:
    def __init__(self, oracle: JaxFunc, *args, **kwargs)-> None:
        """
        oracle means the function oracle.
        """
    @abstractmethod
    def step(self, *args, **kwargs) -> float:
        """
        makes step of optimization and returns value of
        a function at point with guarantees of convergence
        """
        raise NotImplementedError()
    @abstractmethod
    def bounds(self) -> float:
        """
        returns the confidence bounds of function value
        """
        raise NotImplementedError()


class SGMTripleAveraging(BaseOptimizer):
    """Implementation of "Subgradient Method with Triple Averaging",
    p.930 of http://link.springer.com/article/10.1007/s10957-014-0677-5

    optimizer for nonsmooth convex functions
    TODO: взято из другой репы
    """

    def __init__(self, oracle: JaxFunc, projection_function, gamma, L, G, dimension=0):
        """
        L: || grad f|| <= L
        G: ||x - y|| <= G
        """
        self.L = L
        self.G = G
        self.oracle_calls = 0
        self.iteration_number = 0
        self.gamma = gamma

        self.oracle = oracle
        self.projection_function = projection_function

        self.f_k = 0

        if dimension == 0:
            self.lambda_k = self.projection_function(0)
            self.dimension = len(self.lambda_k)
        else:
            self.dimension = dimension
            self.lambda_k = self.projection_function(
                np.zeros(self.dimension, dtype=float)
            )

        self.lambda_0 = copy.deepcopy(self.lambda_k)
        self.x_k = None

        self.s_k = np.zeros(
            self.dimension, dtype=float
        )  # this stores \sum_{k=0}^t diff_d_k

        # for record keeping
        self.method_name = "TA"
        self.parameter = gamma

    def step(self, *args, **kwargs):
        self.x_k, self.f_k, diff_d_k = (
            self.lambda_k,
            self.oracle(self.lambda_k),
            self.oracle.grad(self.lambda_k),
        )

        self.oracle_calls += 1

        # step 1
        self.A_t = self.oracle_calls + 1

        self.s_k += diff_d_k
        gamma_t = self.gamma * np.sqrt(self.iteration_number + 1)
        gamma_t_plus_1 = self.gamma * np.sqrt(self.iteration_number + 2)
        tau_t = float(1) / float(self.iteration_number + 2)

        # lambda_k_plus = float(1.0) / float(gamma_t) * self.s_k  # old
        lambda_k_plus = float(self.A_t) / float(gamma_t) * self.s_k  # fixed

        lambda_k_plus = self.projection_function(lambda_k_plus)

        # step 2
        lambda_k_hat = (
            float(gamma_t) / float(gamma_t_plus_1) * lambda_k_plus
            + (1 - float(gamma_t) / float(gamma_t_plus_1)) * self.lambda_0
        )

        # step 4
        self.lambda_k = (1 - tau_t) * self.lambda_k + tau_t * lambda_k_hat

        self.iteration_number += 1

        return self.f_k

    def bounds(self):
        return self.L * self.G / (self.iteration_number**0.5)


class AcceleratedGradDescent(BaseOptimizer):
    """
    optimizer for smooth convex functions

    """

    def __init__(self, oracle: JaxFunc, x0, L, R):
        """
        L smoothness
        R = ||x0 - x*||
        """
        self.oracle = oracle
        self.x = x0
        self.y = x0
        self.steps = 0
        self.L = L
        self.R = R

    def step(self):
        x_old = self.x
        self.x = self.y - 1 / self.L * self.oracle.grad(self.y)
        self.y = self.x + self.steps / (self.steps + 3) * (self.x - x_old)
        self.steps += 1

        val = self.oracle(self.x)
        return val

    def bounds(self):
        return 2 * self.L * self.R**2 / (self.steps**2 + 5 * self.steps + 6)

class StochasticAGD(BaseOptimizer):
    def __init__(self, oracle: JaxFunc, projection, x0, sigma, L, D, M, gamma=None):
        """
        L, M: f(y)-f(x)-⟨f'(x),y - x⟩≤L∥y-x∥2+M∥y-x∥
        sigma: E ||G - g||^2 \\ leq sigma^2
        D: diameter

        this algo is for optimization on compact

        M <= L * D
        """
        self.x = x0
        self.x_low = x0
        self.x_up = x0

        self.oracle = oracle
        self.projection = projection

        self.steps = 0

        self.sigma, self.M, self.D, self.L = sigma, M, D, L
        if gamma is None:
            gamma = ((M**2 + sigma**2) / D) ** 0.5
        self.gamma = gamma

    def step(self, *args, **kwargs):
        """
        xt_low =(1-qt)x_tm1_up +qt x_tm1,    4.2.5
        xt = argmin{γ_t [⟨G(xt,ξt),x⟩+μV(x_t_low,x)]+V(x_tm1,x)}, x∈X  4.2.6
        x_t_up = (1-α_t)x_tm1_up +α_tx_t.

        α_t = q_t , mu = 0:

        xt_low =(1-α_t)x_tm1_up +α_t x_tm1,    4.2.5
        xt = argmin{γ_t [⟨G(xt,ξt),x⟩]+V(x_tm1,x)}, x∈X  4.2.6
        x_t_up = (1-α_t)x_tm1_up +α_t x_t.

        Lan. Proposition 4.5
        """

        self.steps += 1

        t = self.steps
        alpha_t = 2 / (t + 1)
        gamma_t = (2 * self.L / t + self.gamma * t**0.5) ** (-1)

        # step 1
        self.x_low = (1 - alpha_t) * self.x_up + alpha_t * self.x

        # step 2
        grad_ = self.oracle.grad(self.x_low)
        self.x = self.projection(gamma_t * grad_, self.x)

        # step 3
        self.x_up = (1 - alpha_t) * self.x_up + alpha_t * self.x

        val = self.oracle(self.x_up)
        return val

    def bounds(self):
        t = self.steps
        bound = (
            4 * self.L * self.D / (t * (t + 1))
            + 2 * self.gamma * self.D / t**0.5
            + 1.9 * (self.M**2 + self.sigma**2) / (self.gamma * t**0.5)
        )
        return bound

# class StochasticMR:
#     def __init__(self, oracle: JaxFunc, projection, x0, sigma, M, Dsq):
#         """
#         sigma: E ∥G(x,ξt)- f`(x)∥2_* ≤ sigma^2
#         M: ∥g(x)∥∗≤ M
#         Dsq: D2X ≡ D2X,ν := max V(x1,x)  # diameter of the set
#         """
#         self.y = x0
#         self.x = x0
#         self.projection = projection
#         self.oracle = oracle

#     def step(self, *args, **kwds):
#         # xt+1 = argminx∈Xγt⟨Gt,x⟩+V(xt,x),t = 1,2,...
#         y_old = self.y
#         grad = self.oracle.grad(y_old)
#         grad = gamma_t * grad
#         self.y = self.projection(grad, y_old)

#         gamma_sum_tp1 = gamma_sum_t + gamma_t
#         self.x = 1 / gamma_sum_tp1 * (gamma_sum_t * self.x + gamma_t * self.y)

#         val = self.oracle(self.x)
#         return val

#     def bounds(self):
#         raise NotImplemented
