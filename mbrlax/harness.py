from mbrlax.metrics import Metrics

class ExperimentHarness():
    def __init__(
        self,
        agent,
        env,
        train_episodes, 
        eval_episodes,
        eval_every
    ):
        self.agent = agent
        self.env = env
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes

    def run(self):
        train_metrics = Metrics(agent=self.agent)
        real_collect_driver = Driver(
            env=self.env,
            policy=self.agent.policy,
            observers=[self.agent.real_replay_buffer.add_batch] + train_metrics)
        
        for episode in range(self.train_episodes):
            real_collect_driver.run()
            self.agent.train_model()
            # self.agent.train_policy()
