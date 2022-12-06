from mbrlax.utils import EpisodeMetrics, Driver
import jax


class ExperimentHarness():
    """
    Main training loop with policy evaluation and metrics logging.
    """

    def __init__(self,
                 logger,
                 logging_file,
                 agent,
                 env,
                 max_train_episodes=10,
                 max_eval_episodes=1,
                 num_random_policy_episodes=1,
                 ):
        """
        Args:
          logger,
          logging_file,
          agent,
          env,
          max_train_episodes=10,
          max_eval_episodes=1,
          num_random_policy_episodes
        """

        self.logger = logger
        self.logging_file = logging_file
        self.agent = agent
        self.env = env
        self.max_train_episodes = max_train_episodes
        self.max_eval_episodes = max_eval_episodes
        self.num_random_policy_episodes = num_random_policy_episodes

    def run(self, seed, policy_params):
        """
        Loop across episodes.

        Each episode consists of the following:

        1. collect real experience by rolling out the policy in the environment
        2. store experience
        3. agent training:
          a. train model from scratch on all collected experience
          b. update policy based on simulated experience using the model learned at 3.a
        4. evaluate policy by generating multiple trajectories in the real environment

        """

        key = jax.random.PRNGKey(seed)

        real_driver = Driver(
            env=self.env,
            policy=self.agent.policy,
            max_steps=31,
            num_rollouts=1
        )

        for episode in range(self.max_train_episodes):
            if episode == self.num_random_policy_episodes:
                mode = "collect"

            key, agent_key, driver_key = jax.random.split(key, 3)

            _, experience = real_driver.run(
                key=driver_key,
                policy_params=policy_params,
                mode="random"
            )

            self.agent.replay_buffer.add_batch(experience)

            policy_params = self.agent.train(agent_key, policy_params)

            # TODO: evaluate policy

    def evaluate(self):
        """
        Computes the expected policy performance.

        """
        pass