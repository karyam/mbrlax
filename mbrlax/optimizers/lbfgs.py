from scipy.optimize import minimize
from typing import Callable


class LBFGS:
    def __init__(self, value_and_grads:Callable):
        self.value_and_grads = value_and_grads
        self.pack_params = pack_params
    
    def minimize(self, loss_closure, variables):
        return minimize(
            fun=self.value_and_grads(loss_closure, variables),
            x0=pack_params(jnp.array([1.0, 1.0]), X_m),
            method='L-BFGS-B',
            jac=True
        )
