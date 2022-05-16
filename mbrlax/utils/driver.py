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

    #TODO: more reliable differentiation between jax and non-jax env
    def run(self, key, policy_params, mode):
        reset_key, step_key = jax.random.split(key)
        if mode == "plan": 
            initial_time_step = self.env.reset(reset_key)
        else: initial_time_step = self.env.reset()
        
        def policy_step(carry, tmp):
            key, time_step, mode = carry
            key, policy_key, env_key = jax.random.split(key, 3)
            action = self.policy.step(key, time_step, mode)
            if mode == "plan": 
                next_time_step = self.env.step(key, action)
            else: next_time_step = self.env.step(action)
            carry, y = [key, next_time_step, mode], [time_step, action, next_time_step]
            return carry, y

        _, experience = jax.lax.scan(
            policy_step,
            [step_key, initial_time_step, "plan"],
            [jnp.zeros((self.num_env_steps, 3))],
        )
        
        rewards = jnp.array([step[2].reward for step in experience])
        assert(rewards.size == self.num_env_steps)
        return experience, jnp.sum(rewards)
