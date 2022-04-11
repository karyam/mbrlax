
NoneType = type(None)

class PathwiseSampling(InferenceStrategy):
    def _init_(
        self,
        initial_state_distribution_model,
        batch_size: int
    ):
        self.batch_size = batch_size
        super().__init__(
            initial_state_distribution_model,
        )

    def _euler_maruyama(self, state:tf.Tensor, dx_dt:tf.Tensor, sqrt_cov:tf.Tensor, dt:float) -> tf.Tensor:
        """
        Euler-Maruyama method for approximately solving SDEs.
        """
        _x = x + dt * dx_dt
        if sqrt_cov is None:
            return _x

        rvs = tf.random.normal(_x.shape, dtype=_x.dtype)
        return _x + tf.linalg.matvec((dt ** 0.5) * sqrt_cov, rvs)

    def propagate_policy(self, policy_model, obs):
        return policy_model(obs)
    
    def initial_state(self) -> Union[dict, tf.Tensor]:
        return self.initial_state_distribution_model.sample(sample_shape=[batch_size])
    
    def encode_state(
        self,
        state: tf.Tensor,
        encoder: Union[StateEncoder, NoneType]
    ) -> Union[GaussianMoments, tf.Tensor]:
        if encoder is not None: return encoder(state)
        return state
        
    def step(
        self,
        forward: Callable,
        dt: float,
        state: Union[dict, tf.Tensor]
    ) -> Union[dict, tf.Tensor]:
        """Compute the next state sample."""
        dx_dt, sqrt_cov = forward(t, x)
        return self._euler_maruyama(state=state, dx_dt=dx_dt, sqrt_cov=sqrt_cov, dt=dt)

    def forward(
        state: Union[GaussianMoments, tf.Tensor],
        transition_model: TransitionModel,
        noise: Union[Callable, NoneType],
        policy: TFPolicy,
        encoder: Union[Callable, NoneType]
    ) -> Union[
        Tuple[GaussianMatch, Union[GaussianMatch, NoneType]],
        Tuple[tf.Tensor, Union[tf.Tensor, NoneType]]]:
        
        """
        Compute next state sample after propagating the current
        sample trough encoder, policy and transition_model
        """
        e = state if (encoder is None) else encoder(state)
        eu = e if (policy is None) else tf.concat([e, policy(e)], axis=-1)
        return transition_model.step(eu), None if (noise is None) else noise(e)

    def initial_condiditons(self, transition_model:TransitionModel):
        _paths = transition_model.generate_paths(num_samples=state.shape[0],
                                           num_bases=num_bases,
                                           sample_axis=0)
        transition_model.set_temporary_paths(_paths)