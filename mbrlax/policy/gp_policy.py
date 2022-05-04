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
        objective,
        optimizer,
        inference_strategy
    ):
        self.action_space = action_space
        self.gp_model_spec = gp_model_spec
        self.objective = objective
        self.optimizer = optimizer
        self.inference_strategy = inference_strategy
        self.model = None

    def get_gp_data(self, experience, mode):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = experience
        inputs, targets, rewards, discounts = obs_tm1, a_tm1, r_t, discount_t
        if self.inference_strategy.encoder is not None:
            if mode == "init":
                inputs = self.inference_strategy.encoder(inputs)
            if mode == "train":
                inputs = self.inference_strategy.propagate_encoder(inputs)
        return inputs, targets, rewards, discounts

    def initialize(self, experience):
        inputs, targets, _, _ = self.get_gp_data(experience, mode="init")
        self.model = initialize_gp_model(
            data=(inputs, targets), 
            model_spec=self.gp_model_spec
        )
    
    #TODO: refactor to be used as arg to jax.value_and_grad
    def loss_function(self, observations, rewards, discounts, compile=True):
        def closure() -> tf.Tensor:
            return tf.foldl(lambda sum, obs: sum + self.objective(obs), observations)
        if compile: closure = tf.function(closure)
        return closure

    def step(self, time_step, mode) -> Union[tf.Tensor, GaussianMatch]:
        if mode == "random": return self.action_space.sample()
        if mode == "collect": return self.model(time_step)
        if mode == "plan":
            return self.inference_strategy.propagate_policy(
                obs=time_step.observation,
                policy_model=self.model
            )

    def train(self, experience):
        inputs, targets, rewards, discounts = self.get_gp_data(experience, mode="train")
        variables = self.model.trainable_variables
        loss = self.loss_closure(
            observations=inputs,
            rewards=rewards,
            discounts=discounts
        )
        return self.optimizer.minimize(loss, variables)
        


    
