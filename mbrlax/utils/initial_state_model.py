import abc
import tensorflow as tf

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

    def sample(self, seed, batch_size=1):
        return self.distribution.sample(seed=seed, sample_shape=batch_size)

# class MomentsInitialStateModel(InitialStateModel):
#     def __init__(self, distribution):
#         super().__init__(distribution)
    
#     def sample(self, batch_size=1) -> GaussianMatch:
#         #TODO: think anbout representing the GaussianMoments object as tensors/arrays
#         #TODO: verify gpflow_pilco moment_matching supports batches of moments
#         mx, Sxx = [], []
#         for _ in range(batch_size):
#             mx.append(tf.cast(self.distribution.mean(), default_float()))
#             Sxx.append(tf.cast(self.distribution.covariance(), default_float()))
#         mx, Sxx = tf.stack(mx), tf.stack(Sxx)
#         #TODO: understand what centered=True/False means
#         x = GaussianMoments((mx, Sxx), centered=True)
#         return GaussianMatch(x=x, y=None, cross=None)