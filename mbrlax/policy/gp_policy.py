from typing import *
import numpy as np
import tensorflow as tf
from gpflow_pilco.moment_matching import GaussianMoments, GaussianMatch
from mbrlax.models.initializers import initialize_gp_model

#TODO: refactor to allow for arbitrary number of initial random policy collecion steps
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
        self.gp_model = None

    def get_gp_data(self, experience, flatten=True):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = experience
        inputs, targets, rewards, discounts = obs_tm1, a_tm1, r_t, discount_t
        
        if flatten is True:
            inputs = tf.reshape(inputs, (-1, inputs.shape[-1]))
            targets = tf.reshape(targets, (-1, targets.shape[-1]))
            rewards = tf.reshape(rewards, (-1, rewards.shape[-1]))
            discounts = tf.reshape(discounts, (-1, discounts.shape[-1]))
        
        return inputs, targets, rewards, discounts
    
    #TODO: refactor to be used as arg to jax.value_and_grad
    def loss_closure(self, states, rewards, discounts, compile=True):
        def closure() -> tf.Tensor:
            return tf.foldl(lambda sum, state: sum + self.objective(state), states)
        if compile: closure = tf.function(closure)
        return closure

    def action(self, time_step) -> Union[tf.Tensor, GaussianMatch]:
        if self.gp_model is None:
            return self.action_space.sample()
        return self.inference_strategy.propagate_policy(
            obs=time_step.observation,
            policy_model=self.gp_model
        )

    def train(self, experience):
        # initialise the GP if None
        inputs, targets, rewards, discounts = self.get_gp_data(experience)
        if self.gp_model is None:
            self.gp_model = initialize_gp_model(
                data=(inputs, targets), 
                model_spec=self.gp_model_spec
            )
        # update GP parameters to minimize loss
        variables = self.gp_model.trainable_variables
        loss = self.loss_closure(
            states=inputs,
            rewards=rewards,
            discounts=discounts
        )
        return self.optimizer.minimize(loss, variables)
        


    
