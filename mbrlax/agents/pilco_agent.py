from mbrlax.transition_model import GPTransitionModel
from mbrlax.utils import EpisodeMetrics, Driver, EnvironmentModel, ReplayBuffer

class PilcoAgent():
    def __init__(
        self,
        transition_model,
        reward_model,
        initial_state_model,
        policy,
        policy_training_iterations=1
    ):
        self.environment_model = EnvironmentModel(
            transition_model=transition_model,
            reward_model=reward_model,
            initial_state_model=initial_state_model
        )

        self.policy = policy
        self.policy_training_iterations = policy_training_iterations

        self.real_replay_buffer = ReplayBuffer()
        self.virtual_replay_buffer = ReplayBuffer()
        
        self.virtual_collect_driver = Driver(
            env=self.environment_model,
            policy=self.policy,
            transition_observers=[self.virtual_replay_buffer.add_batch], 
            max_steps=31
        )

    def train_policy(self):
        #INFO: the policy is trained with virtual experience
        result = []
        for it in range(self.policy_training_iterations):
            initial_time_step = self.environment_model.reset()
            self.virtual_collect_driver.run(initial_time_step)
            # get the most recent batch of virtual experience
            virtual_experience = self.virtual_replay_buffer.get_last_n(
                self.virtual_collect_driver.max_steps)
            result.append(self.policy.train(virtual_experience))
        return result

    def train_model(self):
        #INFO: the model is trained with real experience
        # collected inside the main loop: harness.run
        real_experience = self.real_replay_buffer.gather_all()
        result = self.transition_model.train(real_experience)