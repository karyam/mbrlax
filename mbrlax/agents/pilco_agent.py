from mbrlax.transition_model import GPTransitionModel
from mbrlax.utils import EpisodeMetrics, Driver, EnvironmentModel, ReplayBuffer

#TODO: meaningfully log training result
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
            mode = "plan",
            env=self.environment_model,
            policy=self.policy,
            transition_observers=[self.virtual_replay_buffer.add_batch], 
            max_steps=31
        )

    def train_policy(self):        
        if self.policy.model is None:
            real_experience = self.real_replay_buffer.gather_all()
            self.policy.initialize(real_experience)
        
        result = []
        for it in range(self.policy_training_iterations):
            initial_time_step = self.environment_model.reset()
            self.virtual_collect_driver.run(initial_time_step)
            virtual_experience = self.virtual_replay_buffer.get_last_n(
                self.virtual_collect_driver.max_steps)
            result.append(self.policy.train(virtual_experience))
        return result

    def train_model(self):
        real_experience = self.real_replay_buffer.gather_all()
        
        if self.environment_model.transition_model.model is None \
            or self.environment_model.transition_model.reinitialize:
            self.environment_model.transition_model.initialize(real_experience)
        
        result = self.environment_model.transition_model.train(real_experience)

    def train(self):
        print("Training dynamics...")
        self.train_model()
        print("Training policy...")
        self.train_policy()