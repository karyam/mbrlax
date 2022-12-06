from gpjax.datasets import CustomDataset, NumpyLoader
import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
import optax

class SGD:
    def __init__(
        self, 
        optimizer, 
        num_epochs=900,
        callback=None
    ):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.callback = callback
        
    def minimize(self, key, params, objective):
        opt_state = self.optimizer.init(params)
        loss_history = []

        @jax.jit
        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(objective, argnums=1)(key, params)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value
                
        for epoch in range(self.num_epochs):
            params, opt_state, loss_value = step(params, opt_state)
            loss_history.append(loss_value)
            if self.callback: self.callback(epoch, loss_value)

        return loss_history, params
