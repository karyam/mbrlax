class Driver():
    def __init__(
        self,
        env, 
        policy, 
        observers, 
        state_transform, 
        max_steps, 
        max_episodes
    ):
        self.env = env
        self.policy = policy
        self.observers = observers
        self.state_transform = state_transform
        self.max_steps = max_steps
        self.max_episodes = max_episodes

    def run(self, time_step):
        trajectory = []
        while step < self.max_steps and episode < self.max_episodes:
            action = self.policy.action(time_step)
            next_time_step = self.env.step(time_step)
            next_time_step.observation = self.state_transform(next_time_step.observation)
            trajectory.append(Transition(time_step, action, next_time_step))
            time_step = next_time_step

        for observer in self.observers:
            observer(trajectory)
            