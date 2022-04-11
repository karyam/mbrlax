from typing import *
import dm_env
import tensorflow as tf

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
        seed=42
    ):
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.initial_state_model = initial_state_model
        self.termination_model = termination_model
        self.time_step: dm_env.TimeStep
        self.reset_next_step = True
        self.batch_size = batch_size
        self.seed = seed
        self.reset()

    def reset(self) -> dm_env.TimeStep:
        #TODO: ensure no terminal states are sampled for initial_obs
        self.reset_next_step = False
        initial_obs = self.initial_state_model.sample(self.batch_size)
        
        if self.transition_model.inference_strategy.obs_transform is not None:
            initial_obs = inference_strategy.propagate_encoder(initial_obs)
        
        self.time_step = dm_env.restart(initial_obs)
        return self.time_step

    #TODO: check for terminal obs acc to termination model and reset them
    def step(self, action: Any) -> dm_env.TimeStep:
        if self.reset_next_step: return self.reset()
        assert action.shape[0] == self.batch_size #TODO: improve me
        obs = self.time_step.observation
        # get already featurized next obs (if the case applies)
        next_obs = self.transition_model.step(obs, action)
        rewards = self.reward_model(next_obs)
        discount = tf.constant(1.0)
        return dm_env.transition(reward=reward, discount=discount, observation=next_obs)

    #TODO: implement below
    def observation_spec(self):
        pass

    def action_spec(self):
        pass
