import abc

class InferenceStrategy(metaclass=abc.ABCMeta):
    def __init__(self, encoder, noise, dt):
        self.noise = noise
        self.encoder = encoder
        self.dt = dt

    @abc.abstractmethod
    def propagate_policy(self, obs, policy_model):
        pass

    @abc.abstractmethod
    def propagate_model(self, obs, action, model):
        pass

    @abc.abstractmethod
    def propagate_encoder(self, obs):
        pass

    @abc.abstractmethod
    def step(self, obs, action, model):
        pass
