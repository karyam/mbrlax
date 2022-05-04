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
        eval_every=1
    ):
        self.logger = logger
        self.logging_file = logging_file
        self.agent = agent
        self.env = env
        self.max_train_episodes = max_train_episodes
        self.max_eval_episodes = max_eval_episodes
        self.num_random_policy_episodes = num_random_policy_episodes

    def run(self):
        #TODO: clean/general metrics implementation
        episode_metrics = EpisodeMetrics(
            env=self.env,
            agent=self.agent,
            logger=self.logger,
            logging_file=self.logging_file 
        )

        real_collect_driver = Driver(
            mode="random",
            env=self.env,
            policy=self.agent.policy,
            transition_observers=[self.agent.real_replay_buffer.push],
            observers=[episode_metrics],
            max_steps=31
        )
    
        #TODO: use num_random_policy_episodes
        for episode in range(self.max_train_episodes):
            initial_time_step = self.env.reset()
            if episode == self.num_random_policy_episodes:
                driver.mode = "collect"
            if episode >= self.num_random_policy_episodes:
                assert(driver.mode == "collect")
            real_collect_driver.run(initial_time_step)
            self.agent.train()
            #TODO: evaluate policy
