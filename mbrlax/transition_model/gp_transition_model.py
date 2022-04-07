from mbrlax.models import SVGP

class GPTransitionModel():
    def __init__(self, gp_model_spec, inference_strategy, optimizer):
        self.gp_model_spec = gp_model_spec
        self.inference_strategy = inference_strategy
        self.optimizer = optimizer
        self.gp_model = None

    def step(self, obs, action):
        return self.inference_strategy.step(obs, action, self.gp_model)

    def get_gp_input(self, experience):
        pass

    def train(self, experience):
        experience = self.get_gp_input(experience)
        if self.gp_model is None:
            self.gp_model = initialize_gp_model(self.gp_model_spec)
        # compute the loss given experience
        # apply the optimiser on model parameters given loss
        
