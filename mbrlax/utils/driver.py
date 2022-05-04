class Driver():
    def __init__(
        self,
        mode,
        env, 
        policy,
        transition_observers, 
        observers=None,
        max_steps=100, 
        max_episodes=1
    ):
        self.env = env
        self.mode = mode
        
        self.transition_observers = transition_observers
        self.observers = observers

        self.policy = policy
        self.runs_so_far = 0
        self.max_steps = max_steps
        self.max_episodes = max_episodes

    def run(self, time_step):
        step, episode = 0, 0
        while step < self.max_steps and episode < self.max_episodes:
            action = self.policy.step(time_step, self.mode)
            next_time_step = self.env.step(action)
            # if next_time_step.last(): next_time_step = env.reset()

            # push transition to replay buffer
            for observer in self.transition_observers:
                observer((time_step, action, next_time_step))
            
            #TODO: update this to include terminating episodes based on dn_env
            step += 1
            time_step = next_time_step
        self.runs_so_far += 1
        
        # collect step metrics 
        for observer in self.observers: observer(self.runs_so_far)
            