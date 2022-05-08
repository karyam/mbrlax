import jax.numpy as jnp
from gpjax.config import default_float
import jax

def sample_mvn(key, mean, cov, full_cov=False, num_samples=None):
    mean_shape = jnp.array(mean.shape)
    S = jnp.array([num_samples]) if num_samples is not None else jnp.array([1])
    D = jnp.array([mean_shape[-1]])
    leading_dims = jnp.array(mean_shape[:-2])

    if not full_cov:
        # mean: [..., N, D] and cov [..., N, D]
        eps_shape = jnp.concatenate([leading_dims, S, mean_shape[-2:]], axis=0)
        eps = jax.random.normal(key, eps_shape, dtype=default_float())  # [..., S, N, D]
        samples = mean[..., None, :, :] + jnp.sqrt(cov)[..., None, :, :] * eps  # [..., S, N, D]
    else:
        raise NotImplementedError
    
    if num_samples is None:
        return jnp.squeeze(samples, axis=-3)  # [..., N, D]
    return samples  # [..., S, N, D]