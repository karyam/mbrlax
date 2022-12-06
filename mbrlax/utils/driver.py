import jax

# Adapted from https://github.com/RobertTLange/evosax/blob/main/evosax/problems/control_brax.py

class Driver():
    def __init__(
        self,
        env, 
        policy,
        max_steps=100,
        num_rollouts=1,
        param_pop=False,
        param_vmap=None,
        n_devices=None
    ):
        """
        :param param_pop: flag specifying whether rollouts are run for a population of params
        :param param_vmap: dict specifying axes to vmap over for policy params
        """
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.num_rollouts = num_rollouts
        self.param_pop = param_pop
        self.param_vmap = param_vmap
        if n_devices is None: self.n_devices = jax.local_device_count()
        else: self.n_devices = n_devices

        # vmap the rollout function across num_rollouts, and param population dims
        self.rollout_wrapper = jax.vmap(self.rollout, in_axes=(0, None, None))
        if self.param_pop is True:
            self.rollout_wrapper = jax.vmap(self.rollout_wrapper, in_axes=(None, param_vmap, None))
            if self.n_devices > 1:
                self.rollout_wrapper = self.rollout_pmap
                print(
                    f"BraxFitness: {self.n_devices} devices detected. Please make"
                    " sure that the ES population size divides evenly across the"
                    " number of devices to pmap/parallelize over."
                )

    def rollout_pmap(self):
        pass
    
    def run(self, key, policy_params, mode):
        """Placeholder fn for rolling out the policy for multi-evals."""
        assert(mode in ["random", "collect", "plan", "eval"])
        rollout_keys = jax.random.split(key, self.num_rollouts)
        scores, experience, masks = jax.jit(self.rollout_wrapper)(
            rollout_keys, policy_params, mode
        )
        if mode != "eval":
            pass
        return scores, experience

    def rollout(self, key, policy_params, mode):
        """Implements a single policy run in the env."""
        reset_key, step_key = jax.random.split(key)
        state = self.env.reset(reset_key)
        
        def policy_step(carry, tmp):
            key, state, policy_params, cum_reward, valid_mask = carry
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
