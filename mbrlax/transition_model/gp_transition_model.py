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

    def step(self, key, obs, action):
        return self.inference_strategy.step(
            key=key,
            obs=obs,
            action=action,
            model=self.model,
        )

    def format_data(self, experience):
        obs_tm1, a_tm1, _, _, obs_t = experience
        obs = jnp.concatenate([obs_tm1, obs_t[-1, :][None]], axis=0)
        obs = self.inference_strategy.encoder(obs)
        inputs = jnp.concatenate([obs[:-1, :], a_tm1], axis=-1)
        targets = obs[1:, :] - obs[:-1, :]
        return inputs, targets
    
    def initialize(self, experience):
        inputs, targets = self.format_data(experience)
        self.model = initialize_gp_model(
            data=(inputs, targets),
            model_spec=self.gp_model_spec
        )

    def train(self, key, objective):
        inputs, targets = self.format_data(experience)
        return self.optimizer.minimize(
            key=key, 
            params=self.model.trainable_params, 
            objective=objective
        )
