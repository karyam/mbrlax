import abc
import tensorflow as tf
from gpflow_pilco.moment_matching import GaussianMoments
from gpflow.config import default_float

#TODO: how will batches be handled by EnvironmentModel

class InitialStateModel(metaclass=abc.ABCMeta):
    def __init__(self, distribution):
        self.distribution = distribution

    @abc.abstractmethod
    def sample(self, batch_size=1):
        raise NotImplementedError

class ParticleInitialStateModel(InitialStateModel):
    def __init__(self, distribution):
        super().__init__(distribution)

    def sample(self, batch_size=1):
        return self.distribution.sample(sample_shape=batch_size)

class MomentsInitialStateModel(InitialStateModel):
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def sample(self, batch_size=1) -> GaussianMoments:
        #TODO: think anbout representing the GaussianMoments object as tensors/arrays
        #TODO: verify gpflow_pilco moment_matching supports batches of moments
        mx, Sxx = [], []
        for _ in range(batch_size):
            mx.append(tf.cast(self.distribution.mean(), default_float()))
            Sxx.append(tf.cast(self.distribution.covariance(), default_float()))
        mx, Sxx = tf.stack(mx), tf.stack(Sxx)
        #TODO: understand what centered=True/False means
        return GaussianMoments((mx, Sxx), centered=True)