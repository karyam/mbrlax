from typing import Callable, List, Tuple
import gpjax
from gpjax.kernels import Kernel, SeparateIndependent, SquaredExponential
from gpjax.mean_functions import Zero
from gpjax.likelihoods import Gaussian, Likelihood
import gpjax.models as models

from mbrlax.models.initializers import lengthscales_median, inducing_points_kmeans
import jax.numpy as jnp

# Adapted from https://github.com/j-wilson/GPflowPILCO

class SVGP(Callable, models.SVGP):
    def __init__(self, *args, prior: Callable=None, **kwargs):
        self.prior = prior
        super().__init__(*args, **kwargs)

    def __call__(self, x, params):
        return self.predict_f(params=params, Xnew=x)

    @classmethod
    def initialize(
        cls,
        data:Tuple[jnp.ndarray],
        num_inducing: int,
        likelihood: Likelihood=None, 
        mean_function:Callable=None,
        kernels:List[Kernel]=None,
        num_latent_gps:int=None,
        prior:Callable=None
    ):
        X, y = data
        num_latent_gps = num_output_dims = y.shape[1]

        if likelihood is None:
            likelihood = Gaussian()

        if mean_function is None: 
            mean_function = Zero(output_dim=num_output_dims)

        if kernels is None:
            lenghtscales=lengthscales_median(x=X)
            kernels = [SquaredExponential(lengthscales=lenghtscales)] * num_latent_gps
            kernel = SeparateIndependent(kernels=kernels)

        #TODO: allow for independent inducing points once I extend GPJax to handle this
        # inducing_points = [inducing_points_kmeans(x=X, num_inducing=num_inducing)] * num_latent_gps
        inducing_points = inducing_points_kmeans(x=X, num_inducing=num_inducing)

        return cls(
            prior=prior,
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_points,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps
        )