from typing import *
from gpflow_pilco.moment_matching import (
    moment_matching,
    GaussianMatch,
    GaussianMoments,
)
from mbrlax.utils.solvers import MomentMatchingEuler
# from mbrlax.inference_strategy import InferenceStrategy
import tensorflow as tf

class MomentMatchingStrategy():
    def __init__(
        self,
        noise: Callable = None,
        encoder: Callable = None,
        dt: float = 1.0
    ):
        super().__init__(encoder=encoder, noise=noise, dt=dt)

    def propagate_encoder(self, obs:GaussianMatch) -> GaussianMatch:
        return moment_matching(obs.x, self.encoder)

    def propagate_model(self, obs: GaussianMoments, action: GaussianMatch, model) -> GaussianMatch:
        x = obs
        match_policy = action
        match_drift = moment_matching(match_policy.joint(), model)

        # Approx. Cov(x, f) by Cov(x, d) Cov(d, d)^{-1} Cov(d, f) where d = (x, u)
        # is Cov(d, f) premultiplied by Cov(d, d)^{-1}?
        if match_drift.cross[1]:
            # try to avoid multiplying by Cov(x, x)^{-1}
            preinv = match_policy.cross[1]
            cross = match_policy.cross_covariance(preinv=preinv)\
                @ match_drift.cross_covariance(preinv=True), preinv
        else:
            cross = match_drift.cross_covariance()[..., :x.mean().shape[-1], :], False

        chain_match_drift = GaussianMatch(x=x, y=match_drift.y, cross=cross)
        match_noise = None if (
            self.noise is None) else moment_matching(x, self.noise)
        return chain_match_drift, match_noise

    def propagate_model_with_encoder(self, obs: GaussianMatch, action: GaussianMatch, model) -> GaussianMatch:
        x = obs.y
        match_encoder = obs
        match_policy = action
        noise = self.noise
        match_drift = moment_matching(match_policy.joint(), model)

        # Get shape and partition info
        ndims_x = x.mean().shape[-1]
        ndims_u = match_policy.y[0].shape[-1]
        ndims_b = ndims_x - len(self.encoder.active_dims)
        active, inactive = self.encoder.get_partition_indices(ndims_x)

        # Approx. Cov(a, u) by Cov(a, e) Cov(e, e)^{-1} Cov(e, u) where e = encoder(x)
        if match_encoder.cross[1]:  # is Cov(x, e) premultiplied by Cov(x, x)^{-1}?
            Sax = tf.gather(x.covariance(dense=True), active, axis=-2)
            Sae = Sax @ match_encoder.cross_covariance(preinv=True)
        else:
            Sxe = match_encoder.cross_covariance(dense=True)
            Sae = tf.gather(Sxe, active, axis=-2)
        Sau = Sae @ match_policy.cross_covariance(preinv=True)

        # Approx. Cov(x, f) by Cov(x, d) Cov(d, d)^{-1} Cov(d, f) where d = (e, u)
        _, perm = zip(*sorted(zip(active + inactive, range(ndims_x))))
        Sad = tf.concat([Sae, Sau], axis=-1)
        Sbd = match_drift.x.covariance()[..., -ndims_b - ndims_u: -ndims_u, :]
        Sxd = tf.gather(tf.concat([Sad, Sbd], axis=-2), perm, axis=-2)
        Sxf = Sxd @ match_drift.cross_covariance(preinv=True)
        chain_match_drift = GaussianMatch(x=x, y=match_drift.y, cross=(Sxf, False))

        if noise is None:
            chain_match_noise = None
        else:
            preinv = match_encoder.cross[1]
            match_noise = moment_matching(match_encoder.y, noise)
            # Approx. Cov(x, z) by Cov(x, e) Cov(e, e)^{-1} Cov(e, z) where z is noise
            Sxz = match_encoder.cross_covariance(preinv=preinv)\
                @ match_noise.cross_covariance(preinv=True)
            chain_match_noise = GaussianMatch(x=x, y=match_noise.y, cross=(Sxz, preinv))
        return chain_match_drift, chain_match_noise


    def propagate_policy(self, obs: GaussianMatch, policy_model):
        return moment_matching(obs.y, policy_model)

    def step(
        self,  
        obs: GaussianMatch, 
        action: GaussianMatch,
        model
    ) -> GaussianMatch:

        # propagate obs and action trought the dynamics model
        if self.encoder is not None:
            match_drift, match_noise = self.propagate_model_with_encoder(
                obs=obs, action=action, model=model)
        else: match_drift, match_noise = self.propagate_model(
            obs=obs.x, action=action, model=model)

        # apply Euler Maruyama to discretize obs moments
        next_obs = MomentMatchingEuler.step(
            match_drift=match_drift, 
            match_noise=match_noise, 
            dt=self.dt,
            x=obs.x, #if no encoder obs.x will be null
        )

        next_obs = GaussianMatch(x=None, y=None, cross=None)
        
        # propagate obs trough encoder if any
        if self.encoder is not None: 
            next_obs = self.propagate_encoder(obs=next_obs)

        return next_obs
