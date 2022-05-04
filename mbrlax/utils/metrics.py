import numpy as np

#Adapted from https://github.com/j-wilson/GPflowPILCO

#TODO: add more metrics, make this class env independent
class EpisodeMetrics:
    def __init__(self, env, agent, logger, logging_file, episode_length=31):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.logging_file = logging_file
        self.episode_length = episode_length

    def rewards(self, states):
        inference_strategy = self.agent.environment_model.transition_model.inference_strategy
        if inference_strategy.obs_transform is not None:
            feats = self.agent.inference_strategy.encoder(states)
        else: feats = states
        return -self.agent.objective(feats)

    def success(
        self,
        states: np.ndarray,
        radius: float = None,
        prox_threshold: float = 0.2,
        num_consecutive: int = 10,
    ):
        if radius is None: radius = self.env.pole.height
        x, y = self.env.get_tip_coordinates(states)
        prox = np.sqrt(x ** 2 + (y - radius) ** 2) < (prox_threshold * radius)

        for _, group in filter(lambda kg: kg[0] == 1, groupby(prox)):
            if sum(1 for _ in group) >= num_consecutive:
                return True
        return False

    def __call__(self, step):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = self.agent.real_replay_buffer.get_last_n(self.episode_length)
        # print(f'Actions shape: {obs_tm1}')

        # rewards = self.rewards(obs_tm1 + obs_t[-1])
        # success_status = self.success(obs_tm1 + obs_t[-1])

        # metrics_logs = f"Roun {step} metrics: rewards={np.sum(rewards)}, success={success_status}."

        # self.logger.info(metrics_logs)
        # self.logging_file.write(metrics_logs + "\n")
