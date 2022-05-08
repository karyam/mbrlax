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
        batch_size=60, 
        num_epochs=900,
        train_step=None, 
        callback=None
    ):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_step = train_step
        self.callback = callback
        
    def load_data(self, data):
        x, y = data
        training_data = CustomDataset(x, y)
        return NumpyLoader(
            training_data,
            batch_size=min(self.batch_size, x.shape[0]),
            shuffle=True
        )

    def set_train_step(self, train_step):
        self.train_step = train_step

    def minimize(self, params, data):
        train_dataloader = self.load_data(data)
        opt_state = self.optimizer.init(params)
        loss_history = []
        
        for epoch in range(self.num_epochs):
            for batch in train_dataloader:
                loss, updates, opt_state = self.train_step(epoch, params, opt_state, batch)
                params = optax.apply_updates(params, updates)
                loss_history.append(loss)
            
            self.callback(epoch, loss_history)

        return loss_history