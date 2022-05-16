from typing import *
import dm_env

#TODO: add functionality for batched data; what happens in this case? 
#TODO: implement abstract class for reward_model
#TODO: implement discount schedule
class EnvironmentModel(dm_env.Environment):
    def __init__(
        self,
        transition_model,
        reward_model,
        initial_state_model,
        termination_model=None,
        time_step=None,
        batch_size=1,
    ):
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.initial_state_model = initial_state_model
        self.termination_model = termination_model
        self.time_step: dm_env.TimeStep
        self.reset_next_step = True
        self.batch_size = batch_size
        self.reset()

    def reset(self, key) -> dm_env.TimeStep:
        #TODO: ensure no terminal states are sampled for initial_obs
        self.reset_next_step = False
        initial_obs = self.initial_state_model.sample(self.batch_size)
        
        if self.transition_model.inference_strategy.encoder is not None:
            initial_obs = self.transition_model.inference_strategy.propagate_encoder(initial_obs)
        
        self.time_step = dm_env.restart(initial_obs)
        return self.time_step

    #TODO: check for terminal obs acc to termination model and reset them
    def step(self, key, action: Any) -> dm_env.TimeStep:
        if self.reset_next_step: return self.reset()
        obs = self.time_step.observation
        next_obs = self.transition_model.step(key, obs, action)
        rewards = self.reward_model(next_obs)
        discount = tf.constant(1.0)
        return dm_env.transition(reward=reward, discount=discount, observation=next_obs)
