from typing import Callable
from mbrlax.utils import ParticleEuler
from mbrlax.inference_strategy import InferenceStrategy
from mbrlax.utils import sample_mvn
import jax

class ConditionalSamplingStrategy(InferenceStrategy):
    def __init__(self,
        key,
        sampling_strategy,
        noise: Callable = None,
        encoder: Callable = None,
        dt: float = 1.0,
    ):
        self.key = key
        self.sampling_strategy = sampling_strategy
        super().__init__(encoder=encoder, noise=noise, dt=dt)

    def propagate_policy(self, obs, policy_model):
        return policy_model(obs)
    
    def propagate_model(self, obs, action, model):
        model_input = jnp.concatenate([obs, action], axis=-1)
        mean, cov = model(model_input)
        self.key, subkey = jax.random.split(self.key)
        samples = self.sampling_strategy(subkey, mean, cov)
        return samples, None if (self.noise is None) else self.noise(samples)

    def propagate_encoder(self, obs):
        return self.encoder(obs)
        
    def step(self, obs, action, model):
        dx_dt, sqrt_cov = self.propagate_model(obs, action, model)
        self.key, subkey = jax.random.split(self.key)
        next_obs = ParticleEuler.step(
            key=subkey, 
            dt=self.dt, 
            x=obs, 
            dx_dt=dx_dt, 
            sqrt_cov=sqrt_cov
        )
        if self.encoder: next_obs = self.propagate_encoder(next_obs)
        return next_obs