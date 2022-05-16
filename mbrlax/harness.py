from mbrlax.utils import EpisodeMetrics, Driver

#TODO: implement clean hyperparameter setup
class ExperimentHarness():
    def __init__(self,
        logger,
        logging_file,
        agent,
        env,
        max_train_episodes, 
        max_eval_episodes,
        num_random_policy_episodes=1,
    ):
        self.logger = logger
        self.logging_file = logging_file
        self.agent = agent
        self.env = env
        self.max_train_episodes = max_train_episodes
        self.max_eval_episodes = max_eval_episodes
        self.num_random_policy_episodes = num_random_policy_episodes

    def run(self, seed):
        key = jax.random.PRNGKey(seed)
        mode = "random"
        real_driver = Driver(
            env=self.env,
            policy=self.agent.policy,
            max_steps=31
        )

        for episode in range(self.max_train_episodes):
            key, agent_key, driver_key = jax.random.split(key, 3)
            if episode == self.num_random_policy_episodes: mode = "collect"
            experience = real_driver.run(
                key=driver_key,
                policy_params=self.agent.policy.trainable_params,
                mode=mode
            )
            self.agent.replay_buffer.add_batch(experience)
            self.agent.train(agent_key)
            #TODO: evaluate policy
