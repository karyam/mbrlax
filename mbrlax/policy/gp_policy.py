from typing import *
import numpy as np
import tensorflow as tf
from gpflow_pilco.moment_matching import GaussianMoments, GaussianMatch
from mbrlax.models.initializers import initialize_gp_model

class GPPolicy:
    def __init__(
        self,
        action_space,
        gp_model_spec,
        optimizer,
        inference_strategy
    ):
        self.action_space = action_space
        self.gp_model_spec = gp_model_spec
        self.optimizer = optimizer
        self.inference_strategy = inference_strategy
        self.model = None

    def format_data(self, experience, mode):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = experience
        inputs, targets, rewards, discounts = obs_tm1, a_tm1, r_t, discount_t
        if self.inference_strategy.encoder is not None:
            if mode == "init":
                inputs = self.inference_strategy.encoder(inputs)
            if mode == "train":
                inputs = self.inference_strategy.propagate_encoder(inputs)
        return inputs, targets, rewards, discounts

    def initialize(self, experience):
        inputs, targets, _, _ = self.format_data(experience, mode="init")
        self.model = initialize_gp_model(
            data=(inputs, targets), 
            model_spec=self.gp_model_spec
        )

    def step(self, key, time_step, mode) -> Union[tf.Tensor, GaussianMatch]:
        if mode == "random": return self.action_space.sample()
        if mode == "collect": return self.model(time_step)
        if mode == "plan":
            return self.inference_strategy.propagate_policy(
                obs=time_step.observation,
                policy_model=self.model
            )

    def train(self, key, policy_params, objective):
        return self.optimizer.minimize(
            key=key,
            params=policy_params, 
            objective=objective
        )
        


    
