from typing import NamedTuple, Type, Callable, List, Optional
import jax.numpy as jnp
from gpjax.likelihoods import Likelihood
from gpjax.config import default_float
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import pdist
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors

from mbrlax.models.utils import KernelRegressor, InverseLinkWrapper

class GPModelSpec(NamedTuple):
    type: Type
    num_inducing: int = 32
    likelihood: Likelihood = None
    prior: Callable = None
    mean_function: Callable = None
    model_uncertainty: bool = False
    kernel: List[Callable] = None
    num_latent_gps: int = None
    invlink: Callable = None

def initialize_gp_model(data, model_spec):
    gp_model = model_spec.type.initialize(
        data=data,
        num_inducing=model_spec.num_inducing,
        likelihood=model_spec.likelihood,
        mean_function=model_spec.mean_function,
        kernels=model_spec.kernel,
        num_latent_gps=model_spec.num_latent_gps,
        prior=model_spec.prior
    )

    if gp_model.q_mu.shape[-2] >= len(data[0]):
        set_trainable(gp_model.inducing_variable, False)

    if not model_spec.model_uncertainty:
        set_trainable(gp_model.q_sqrt, False)
        for kernel in gp_model.kernel.kernels:
            set_trainable(kernel.variance, False)
        gp_model = KernelRegressor(model=gp_model)

    if model_spec.invlink is not None:
        gp_model = InverseLinkWrapper(model=gp_model, invlink=model_spec.invlink)

    return gp_model

def inducing_points_kmeans(
    x,
    num_inducing:int,
    batch_size:int = 1024,
    init: str = "k-means++",
    seed: Optional[int] = None):

    if x.shape[0] < num_inducing: return jnp.array(x)
    kmeans = MiniBatchKMeans(
        n_clusters=num_inducing,
        batch_size=min(x.shape[0], batch_size),
        random_state=seed,
        init=init
    )
    kmeans.fit(x)
    points = kmeans.cluster_centers_
    nmissing = num_inducing - len(points)
    if nmissing > 0:
        rvs = jax.random.normal(shape=(nmissing, points.shape[-1]))
        points = jnp.vstack([points, rvs])
    return points

def lengthscales_median(x, transform=None, lower=0.01, upper=100.0):
    #TODO: update GPJax to allow for custom transform
    # if transform is None:
    #     transform = tfb.Sigmoid(
    #         low=jnp.asarray(lower, dtype=default_float()),
    #         high=jnp.asarray(upper, dtype=default_float())
    #     )
    _lower, _upper = 1.1 * lower, 0.9 * upper
    dist = pdist(x, metric='euclidean')
    init = jnp.full(
        shape=x.shape[-1],
        fill_value=jnp.clip(jnp.sqrt(0.5) * jnp.median(dist), a_min=_lower, a_max=_upper))
    
    return init

