import collections
import numpy as np
from mbrlax.settings import FLAGS
from typing import Any, Callable
from dm_env import TimeStep
import random

# Adapted from https://github.com/deepmind/rlax/blob/master/examples/simple_dqn.py

# TODO: define this inside hyper-params settup
discount_factor = 0.99

#TODO: consider storing batched data
class ReplayBuffer():
    def __init__(self, capacity: int=1e7):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.buffer = []

    def push(self, item):
        time_step, action, next_time_step = item
        if next_time_step.discount is None: discount = 0 
        else: discount = next_time_step.discount
        self.buffer.append((
            time_step.observation,
            action[None],
            next_time_step.reward,
            discount,
            next_time_step.observation
        ))

    def add_batch(self, batch):
        for item in batch: self.push(item)

    def format_data(self, transitions):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(*transitions)
        
        return (
            np.squeeze(np.stack(obs_tm1)), 
            np.asarray(a_tm1), 
            np.asarray(r_t),
            np.asarray(discount_t) * discount_factor, 
            np.squeeze(np.stack(obs_t))
        )


    def sample(self, batch_size: int):
        batch_size = min(len(self.buffer), batch_size)
        random_transitons = random.sample(self.buffer, batch_size)
        return self.format_data(transitions=random_transitions)
        

    def get_last_n(self, n: int):
        last_n_transitions = self.buffer[-n:]
        return self.format_data(transitions=last_n_transitions)

    def gather_all(self):
        return self.format_data(transitions=self.buffer)
