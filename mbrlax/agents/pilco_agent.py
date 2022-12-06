from mbrlax.transition_model import GPTransitionModel
from mbrlax.utils import EpisodeMetrics, Driver, EnvironmentModel, ReplayBuffer
import jax.numpy as jnp
from gpjax.parameters import build_constrain_params
from functools import partial

#TODO: meaningfully log training result
class PilcoAgent():
    def __init__(
        self,
        transition_model,
        reward_model,
        initial_state_model,
        policy,
    ):
        self.environment_model = EnvironmentModel(
            transition_model=transition_model,
            reward_model=reward_model,
            initial_state_model=initial_state_model
        )
        self.policy = policy
        self.replay_buffer = ReplayBuffer()
        self.virtual_driver = Driver(
            env=self.environment_model,
            policy=self.policy,
            max_steps=31
        )

    # @partial(jax.jit, static_argnums=(0,))
    def train_policy(self, key, policy_params):        
        def policy_objective(key, policy_params):
            _, long_term_reward = self.virtual_driver.run(
                key=key,
                policy_params=policy_params,
                mode="plan"
            )
            return long_term_reward
            
        return self.policy.train(
            key=key, 
            policy_params=policy_params,
            objective=policy_objective
        )

    # @partial(jax.jit, static_argnums=(0,))
    def train_model(self, key, experience):
        svgp_transforms = self.environment_model.transition_model.model.get_transforms()
        constrain_params = build_constrain_params(svgp_transforms)
        elbo = self.environment_model.transition_model.model.build_elbo(
        constrain_params=constrain_params, num_data=experience[0].shape[0])

        def model_objective(key, model_params):
            data = self.environment_model.transition_model.format_data(experience)
            return - elbo(model_params, data)
        
        result = self.environment_model.transition_model.train(key=key, objective=model_objective)
        return result

    def train(self, key, policy_params):
        model_key, policy_key = jax.random.split(key)
        experience = self.replay_buffer.gather_all()

        if self.environment_model.transition_model.model is None \
            or self.environment_model.transition_model.reinitialize:
            print("Initialise dynamics.")
            self.environment_model.transition_model.initialize(experience)
    
        # print("Training dynamics...")
        # self.train_model(model_key, experience)
        
        if self.policy.model is None:
            print("Initialise policy.")
            self.policy.initialize(experience)
        
        print("Training policy...")
        self.train_policy(policy_key)