import jax.numpy as jnp
from mbrlax.models import initialize_gp_model

class GPTransitionModel():
    def __init__(
        self,
        gp_model_spec,
        inference_strategy,
        optimizer,
        reinitialize=True,
        param_transform=None
    ):
        self.gp_model_spec = gp_model_spec
        self.inference_strategy = inference_strategy
        self.optimizer = optimizer
        self.reinitialize = reinitialize
        self.model = None

    def step(self, obs, action):
        return self.inference_strategy.step(
            obs=obs,
            action=action,
            model=self.model,
        )

    def get_gp_data(self, experience):
        obs_tm1, a_tm1, _, _, obs_t = experience
        obs = jnp.concatenate([obs_tm1, obs_t[-1, :][None]], axis=0)
        
        if self.inference_strategy is not None and \
           self.inference_strategy.encoder is not None:
            obs = self.inference_strategy.encoder(obs)
        
        inputs = jnp.concatenate([obs[:-1, :], a_tm1], axis=-1)
        targets = obs[1:, :] - obs[:-1, :]
        return inputs, targets
    
    def initialize(self, experience):
        inputs, targets = self.get_gp_data(experience)
        self.model = initialize_gp_model(
            data=(inputs, targets),
            model_spec=self.gp_model_spec
        )   

    def loss_function(self, num_data):
        return self.model.loss_function_closure(num_data)

    def train(self, experience):
        inputs, targets = self.get_gp_data(experience)
        init_params = self.model.trainable_variables
        loss_function = self.model.loss_function_closure(inputs.shape[0])
        return self.optimizer.minimize(loss_function, init_params, (inputs, targets))
