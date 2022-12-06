
import tensorflow as tf
from gpflow_pilco.moment_matching import GaussianMoments, GaussianMatch
import jax.numpy as jnp

# Adapted from https://github.com/j-wilson/GPflowPILCO

class ParticleEuler:
    @classmethod
    def step(cls, key, dt, x, dx_dt, sqrt_cov):
        """
        Euler-Maruyama method for approximately solving SDEs.
        """
        _x = x + dt * dx_dt
        if sqrt_cov is None: return _x

        rvs = jnp.random.normal(key, _x.shape, dtype=_x.dtype)
        return _x + jnp.matmul((dt ** 0.5) * sqrt_cov, rvs)


class MomentMatchingEuler:
    @classmethod
    def step(
        cls: "MomentMatchingEuler",
        match_drift: GaussianMatch,
        match_noise: GaussianMatch,
        dt: float,
        x: GaussianMoments
    ) -> GaussianMatch:
        """
        Moment matching variant of Euler-Maruyama solver.
        """
        mx = x.mean()
        Sxx = x.covariance()

        mf = match_drift.y.mean()
        Sxf = match_drift.cross_covariance()
        Sff = match_drift.y.covariance()

        _mx = mx + dt * mf
        print(f"Shape Sxx: {Sxx.shape}")
        print(f"Shape (dt ** 2) * Sff: {((dt ** 2) * Sff).shape}")
        print(f"Shape dt * (Sxf + tf.linalg.adjoint(Sxf)): {(dt * (Sxf + tf.linalg.adjoint(Sxf))).shape}")
        _Sxx = Sxx + dt * (Sxf + tf.linalg.adjoint(Sxf)) + (dt ** 2) * Sff
        if match_noise is not None:
            Sxz = match_drift.cross_covariance()
            Szz = match_drift.y.covariance()
            _Sxx += (dt ** 0.5) * (Sxz + tf.linalg.adjoint(Sxz)) + dt * Szz

        return GaussianMoments(moments=(_mx, _Sxx), centered=True)
