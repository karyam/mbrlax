from typing import *
from gpflow_pilco.moment_matching import (
    moment_matching,
    GaussianMatch,
    GaussianMoments,
)
from mbrlax.utils.solvers import MomentMatchingEuler

class MomentMatchingStrategy:
    def __init__(
        self,
        noise: Callable = None,
        obs_transform: Callable = None,
        dt: float = 1.0
    ):
        self.noise = noise
        self.obs_transform = obs_transform
        self.dt = dt

    def propagate_encoder(self, obs:GaussianMatch) -> GaussianMatch:
        return moment_matching(obs, self.obs_transform)

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

    def propagate_model_with_transform(self, obs: GaussianMatch, action: GaussianMatch, model) -> GaussianMatch:
        x = obs.x
        match_encoder = obs
        pass

    def propagate_policy(self, obs: GaussianMatch, policy_model):
        return moment_matching(obs.y, policy_model)

    def step(
        self,  
        obs: GaussianMatch, 
        action: GaussianMatch,
        model
    ) -> GaussianMatch:

        # propagate obs and action trought the dynamics model
        if self.obs_transform is not None:
            match_drift, match_noise = self.propagate_model_with_transform(
                obs=obs, action=action, model=model)
        else: match_drift, match_noise = self.propagate_model(
            obs=obs.y, action=action, model=model)

        # apply Euler Maruyama to discretize obs moments
        next_obs = MomentMatchingEuler.step(
            match_drift=match_drift, 
            match_noise=match_noise, 
            dt=self.dt,
            x=obs.y, #if no obs_transform obs.x will be null
        )
        
        # propagate obs trough transform if any
        if self.obs_transform is not None: return self.propagate_encoder(obs=next_obs)
        else: return GaussianMatch(x=None, y=next_obs, cross=None)
