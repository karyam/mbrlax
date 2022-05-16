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
        grads = jax.jit(jax.value_and_grad(objective))
        for epoch in range(self.num_epochs):
            for batch in train_dataloader:
                loss, grad = grads()
                print(grad)
                updates, opt_state = self.optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                loss_history.append(loss)
            
            self.callback(epoch, loss_history)

        return loss_history