import jax.numpy as jnp

class GaussianObjective:
    def __init__(self, target, precis):
        self.target = target
        self.precis = precis

    def __call__(self, x):
        err = x - self.target
        dist2 = jnp.sum(err * jnp.matmul(self.precis, err), -1)
        return -jnp.exp(-0.5 * dist2)