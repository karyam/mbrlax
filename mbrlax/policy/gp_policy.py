import jax

class GPPolicy:
    def __init__(
        self,
        action_space,
        gp_model_spec,
        optimizer
    ):
        self.action_space = action_space
        self.gp_model_spec = gp_model_spec
        self.optimizer = optimizer
        self.gp_model = None

    def action(self, time_step, key):
        if self.gp_model is None:
            return jax.random.choice(key=key, a=action_space)
