class Driver():
    def __init__(
        self,
        env, 
        policy,
        max_steps=100,
    ):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps

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

    def run(self, key, policy_params, mode):
        
        def policy_step(input, tmp):
            time_step, mode = input
            action = self.policy.step(time_step, mode)
            next_time_step = self.env.step(action)
            carry, y = [next_time_step, mode], [next_time_step]
            return carry, y

        _, experience = jax.lax.scan(
            policy_step,
            [obs, state, policy_params, rng_episode],
            [jnp.zeros((self.num_env_steps, 2))],
        )
        
        
        
        return experience, 
        
            

            