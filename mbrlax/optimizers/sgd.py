from gpjax.datasets import CustomDataset, NumpyLoader
import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
import optax

@jax.jit
def train_step(params, epoch, optimizer, opt_state, batch):
    loss, grads = jax.value_and_grad(LOSS_FUNCTION, argnums=0)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, updates, opt_state

class SGD:
    def __init__(self, optimizer, batch_size=60, num_epochs=900, callback=None):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.callback = callback

    def load_data(self, data):
        x, y = data
        training_data = CustomDataset(x, y)
        return NumpyLoader(
            training_data,
            batch_size=min(self.batch_size, x.shape[0]),
            shuffle=True
        )

    def minimize(self, loss_function, params, data):
        train_dataloader = self.load_data(data)
        opt_state = self.optimizer.init(params)
        loss_history = []
        # value_and_grads = jit(jax.value_and_grad(loss_function, argnums=0))
        
        for epoch in range(self.num_epochs):
            for batch in train_dataloader:
                loss, updates, opt_state = train_step(
                    params=params,
                    epoch=epoch,
                    optimizer=self.optimizer,
                    opt_state=opt_state,
                    batch=batch
                )
                params = optax.apply_updates(params, updates)
                loss_history.append(loss)
            
            self.callback(epoch, loss_history)

        return loss_history