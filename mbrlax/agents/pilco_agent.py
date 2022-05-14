from mbrlax.transition_model import GPTransitionModel
from mbrlax.utils import EpisodeMetrics, Driver, EnvironmentModel, ReplayBuffer
import jax.numpy as jnp

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

        self.replay_buffer = ReplayBuffer()
        
        self.virtual_driver = Driver(
            mode = "plan",
            env=self.environment_model,
            policy=self.policy,
            max_steps=31
        )

    def train_policy(self, key):        
        if self.policy.model is None:
            real_experience = self.real_replay_buffer.gather_all()
            self.policy.initialize(real_experience)

        def objective_closure(policy_params, key):
            experience = self.virtual_collect_driver.run(
                key=key,
                policy_params=policy_params,
                mode="plan"
            )
            rewards, dones = experience
            ep_mask = (jnp.cumsum(dones) < 1).reshape(self.num_env_steps, 1)
            return jnp.sum(rewards * ep_mask)
        
        return self.policy.train(key, objective_closure)

    def train_model(self, key):
        real_experience = self.real_replay_buffer.gather_all()
        
        if self.environment_model.transition_model.model is None \
            or self.environment_model.transition_model.reinitialize:
            self.environment_model.transition_model.initialize(real_experience)
        
        result = self.environment_model.transition_model.train(key, real_experience)

    def train(self, key):
        model_key, policy_key = jax.random.split(key)
        print("Training dynamics...")
        self.train_model(model_key)
        print("Training policy...")
        self.train_policy(policy_key)